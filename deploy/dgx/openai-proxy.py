#!/usr/bin/env python3
"""Routing proxy for multi-model vLLM deployment.

Routes /v1/* requests to the correct vLLM instance based on the model name.
Provides a unified /v1/models listing and health endpoint.
Translates /v1/responses to /v1/chat/completions on the backend and converts
the response back to Responses API format for SDK compatibility.

Zero external dependencies — uses only Python stdlib.
"""

import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any

PORT = 9000

MODEL_ROUTES: dict[str, str] = {
    "qwen38_briefs": "http://localhost:8002",
    # Vision model used to verify the text rendered into a generated image.
    "qwen_vl_ocr": "http://localhost:8003",
}

DEFAULT_MODEL = "qwen38_briefs"
MAX_COMPLETION_TOKENS = 16384


def _clamp_tokens(body: dict) -> dict:
    for key in ("max_completion_tokens", "max_tokens"):
        if isinstance(body.get(key), int) and body[key] > MAX_COMPLETION_TOKENS:
            body[key] = MAX_COMPLETION_TOKENS
    return body


def log(msg: str) -> None:
    print(f"[proxy] {msg}", flush=True)


def _uid() -> str:
    return uuid.uuid4().hex[:16]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            models = []
            for name, url in MODEL_ROUTES.items():
                try:
                    with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as resp:
                        data = json.loads(resp.read())
                        models.extend(data.get("data", []))
                except Exception:  # noqa: BLE001
                    models.append({"id": name, "object": "model", "owned_by": "local"})
            self._json(200, {"object": "list", "data": models})
        elif self.path in ("/health", "/v1/health"):
            healthy = {}
            for name, url in MODEL_ROUTES.items():
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                        healthy[name] = resp.status == 200
                except Exception:  # noqa: BLE001
                    healthy[name] = False
            status = 200 if any(healthy.values()) else 503
            self._json(
                status, {"status": "ok" if status == 200 else "unhealthy", "models": healthy}
            )
        else:
            self._proxy_get()

    def do_POST(self) -> None:
        body = _clamp_tokens(self._read_body())
        if "/responses" in self.path:
            model = body.get("model", DEFAULT_MODEL)
            backend = MODEL_ROUTES.get(model)
            if not backend:
                self._json(400, {"error": {"message": f"Unknown model: {model}"}})
                return
            chat_body = self._responses_to_chat(body)
            is_streaming = chat_body.get("stream", False)
            log(f"POST /v1/responses → /v1/chat/completions model={model} stream={is_streaming}")
            if is_streaming:
                self._responses_streaming(backend, chat_body, body)
            else:
                self._responses_non_streaming(backend, chat_body, body)
            return
        model = body.get("model", DEFAULT_MODEL)
        backend = MODEL_ROUTES.get(model)
        if not backend:
            self._json(
                400,
                {
                    "error": {
                        "message": f"Unknown model: {model}. Available: {list(MODEL_ROUTES.keys())}"
                    }
                },
            )
            return
        is_streaming = body.get("stream", False)
        log(f"POST {self.path} model={model} stream={is_streaming}")
        if is_streaming:
            self._proxy_streaming(backend, body)
        else:
            self._proxy_request(backend, body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    # ── Responses API translation ──────────────────────────────────────

    @staticmethod
    def _responses_to_chat(body: dict) -> dict:
        messages: list[dict[str, Any]] = []
        instructions = body.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": instructions})

        input_items = body.get("input", [])
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.get("call_id", ""),
                                "content": item.get("output", ""),
                            }
                        )
                        continue
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict):
                                text = part.get("text", "")
                                parts.append({"type": "text", "text": text})
                            elif isinstance(part, str):
                                parts.append({"type": "text", "text": part})
                        content = parts
                    messages.append({"role": role, "content": content})

        chat_body: dict[str, Any] = {
            "model": body.get("model", DEFAULT_MODEL),
            "messages": messages,
            "stream": body.get("stream", True),
        }
        if body.get("temperature") is not None:
            chat_body["temperature"] = body["temperature"]
        if body.get("max_output_tokens") is not None:
            chat_body["max_tokens"] = body["max_output_tokens"]
        if body.get("top_p") is not None:
            chat_body["top_p"] = body["top_p"]

        tools = body.get("tools", [])
        if tools:
            chat_tools = []
            for t in tools:
                if isinstance(t, dict) and t.get("type") == "function":
                    chat_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": t.get("name", ""),
                                "description": t.get("description", ""),
                                "parameters": t.get("parameters", {}),
                            },
                        }
                    )
            if chat_tools:
                chat_body["tools"] = chat_tools

        return chat_body

    def _responses_non_streaming(self, backend: str, chat_body: dict, orig: dict) -> None:
        url = f"{backend}/v1/chat/completions"
        data = json.dumps(chat_body).encode()
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                chat_resp = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            return
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": {"message": f"Backend error: {e}"}})
            return

        choice = chat_resp.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        usage = chat_resp.get("usage", {})

        resp_id = f"resp_{_uid()}"
        msg_id = f"msg_{_uid()}"
        output: list[dict[str, Any]] = []

        if content.strip():
            output.append(
                {
                    "type": "message",
                    "id": msg_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": content.strip(), "annotations": []}
                    ],
                }
            )

        for tc in tool_calls:
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{_uid()}",
                    "call_id": tc.get("id", f"call_{_uid()}"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "status": "completed",
                }
            )

        responses_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": chat_resp.get("model", orig.get("model", DEFAULT_MODEL)),
            "output": output,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        resp_body = json.dumps(responses_obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def _responses_streaming(self, backend: str, chat_body: dict, orig: dict) -> None:
        parsed = urllib.parse.urlparse(backend)
        conn = http.client.HTTPConnection(
            parsed.hostname or "localhost", parsed.port or 80, timeout=300
        )
        data = json.dumps(chat_body).encode()
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            conn.request("POST", "/v1/chat/completions", data, headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            log(f"Backend connection error: {e}")
            self._json(502, {"error": {"message": f"Backend error: {e}"}})
            return

        if resp.status != 200:
            resp_body = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        resp_id = f"resp_{_uid()}"
        msg_id = f"msg_{_uid()}"
        model = orig.get("model", DEFAULT_MODEL)
        seq = 0
        sent_preamble = False
        content_started = False
        tool_calls: dict[int, dict[str, Any]] = {}
        accumulated_content = ""

        def _emit(event_type: str, payload: dict) -> None:
            nonlocal seq
            payload["type"] = event_type
            payload["sequence_number"] = seq
            seq += 1
            line = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()

        def _send_preamble() -> None:
            nonlocal sent_preamble
            if sent_preamble:
                return
            sent_preamble = True
            resp_obj = {
                "id": resp_id,
                "object": "response",
                "created_at": int(time.time()),
                "model": model,
                "status": "in_progress",
                "output": [],
            }
            _emit("response.created", {"response": resp_obj})
            _emit("response.in_progress", {"response": resp_obj})
            _emit(
                "response.output_item.added",
                {
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            )
            _emit(
                "response.content_part.added",
                {
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": msg_id,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )

        try:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                decoded = raw_line.decode("utf-8", errors="replace").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                payload = decoded[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                text = delta.get("content")
                if text is not None:
                    _send_preamble()
                    if not content_started and not text.strip():
                        continue
                    content_started = True
                    accumulated_content += text
                    _emit(
                        "response.output_text.delta",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": msg_id,
                            "delta": text,
                        },
                    )

                tc_list = delta.get("tool_calls", [])
                for tc in tc_list:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls:
                        _send_preamble()
                        tc_id = tc.get("id", f"call_{_uid()}")
                        fn = tc.get("function", {})
                        tool_calls[idx] = {
                            "id": f"fc_{_uid()}",
                            "call_id": tc_id,
                            "name": fn.get("name", ""),
                            "arguments": "",
                            "output_index": len(tool_calls) + 1,
                        }
                        _emit(
                            "response.output_item.added",
                            {
                                "output_index": tool_calls[idx]["output_index"],
                                "item": {
                                    "id": tool_calls[idx]["id"],
                                    "type": "function_call",
                                    "call_id": tc_id,
                                    "name": tool_calls[idx]["name"],
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            },
                        )
                    else:
                        fn = tc.get("function", {})
                    args_chunk = fn.get("arguments", "") if "function" in tc else ""
                    if args_chunk:
                        tool_calls[idx]["arguments"] += args_chunk
                        _emit(
                            "response.function_call_arguments.delta",
                            {
                                "output_index": tool_calls[idx]["output_index"],
                                "item_id": tool_calls[idx]["id"],
                                "delta": args_chunk,
                            },
                        )

                if finish:
                    _send_preamble()
                    _emit(
                        "response.output_text.done",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": msg_id,
                            "text": accumulated_content,
                        },
                    )
                    _emit(
                        "response.content_part.done",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": msg_id,
                            "part": {
                                "type": "output_text",
                                "text": accumulated_content,
                                "annotations": [],
                            },
                        },
                    )
                    _emit(
                        "response.output_item.done",
                        {
                            "output_index": 0,
                            "item": {
                                "id": msg_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": accumulated_content,
                                        "annotations": [],
                                    }
                                ],
                            },
                        },
                    )

                    for idx in sorted(tool_calls):
                        tc_info = tool_calls[idx]
                        _emit(
                            "response.function_call_arguments.done",
                            {
                                "output_index": tc_info["output_index"],
                                "item_id": tc_info["id"],
                                "arguments": tc_info["arguments"],
                            },
                        )
                        _emit(
                            "response.output_item.done",
                            {
                                "output_index": tc_info["output_index"],
                                "item": {
                                    "id": tc_info["id"],
                                    "type": "function_call",
                                    "call_id": tc_info["call_id"],
                                    "name": tc_info["name"],
                                    "arguments": tc_info["arguments"],
                                    "status": "completed",
                                },
                            },
                        )

                    output: list[dict[str, Any]] = []
                    if accumulated_content.strip():
                        output.append(
                            {
                                "type": "message",
                                "id": msg_id,
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": accumulated_content,
                                        "annotations": [],
                                    }
                                ],
                            }
                        )
                    for idx in sorted(tool_calls):
                        tc_info = tool_calls[idx]
                        output.append(
                            {
                                "type": "function_call",
                                "id": tc_info["id"],
                                "call_id": tc_info["call_id"],
                                "name": tc_info["name"],
                                "arguments": tc_info["arguments"],
                                "status": "completed",
                            }
                        )

                    _emit(
                        "response.completed",
                        {
                            "response": {
                                "id": resp_id,
                                "object": "response",
                                "created_at": int(time.time()),
                                "model": model,
                                "status": "completed",
                                "output": output,
                            },
                        },
                    )

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except BrokenPipeError:
            log("Client disconnected (broken pipe)")
        finally:
            conn.close()
            log(f"Stream done: {seq} events emitted")

    # ── Pass-through proxy methods ─────────────────────────────────────

    def _proxy_streaming(self, backend: str, body: dict) -> None:
        parsed = urllib.parse.urlparse(backend)
        conn = http.client.HTTPConnection(
            parsed.hostname or "localhost", parsed.port or 80, timeout=300
        )
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            conn.request("POST", self.path, data, headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            log(f"Backend connection error: {e}")
            self._json(502, {"error": {"message": f"Backend error: {e}"}})
            return

        if resp.status != 200:
            resp_body = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                self.wfile.write(raw_line)
                self.wfile.flush()
        except BrokenPipeError:
            log("Client disconnected (broken pipe)")
        finally:
            conn.close()
            log("Stream done")

    def _proxy_request(self, backend: str, body: dict) -> None:
        url = f"{backend}{self.path}"
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key in ("Content-Type",):
                    val = resp.headers.get(key)
                    if val:
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except BrokenPipeError:
            log("Client disconnected (broken pipe)")
        except Exception as e:  # noqa: BLE001
            log(f"Backend error: {e}")
            try:
                self._json(502, {"error": {"message": f"Backend error: {e}"}})
            except BrokenPipeError:
                pass

    def _proxy_get(self) -> None:
        backend = next(iter(MODEL_ROUTES.values()))
        url = f"{backend}{self.path}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for key in ("Content-Type",):
                    val = resp.headers.get(key)
                    if val:
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": {"message": f"Backend error: {e}"}})

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    log(f"vLLM routing proxy on 0.0.0.0:{PORT}")
    log(f"Routes: {MODEL_ROUTES}")
    sys.stdout.flush()
    server = ThreadedServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
