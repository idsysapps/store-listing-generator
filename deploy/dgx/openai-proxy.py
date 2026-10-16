#!/usr/bin/env python3
"""Routing proxy for multi-model vLLM deployment.

Routes /v1/* requests to the correct vLLM instance based on the model name.
Provides a unified /v1/models listing and health endpoint.
Converts Qwen3 <think> tokens into reasoning_content for OpenAI-compatible clients.

Zero external dependencies — uses only Python stdlib.
"""

import http.client
import json
import re
import sys
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

PORT = 9000

MODEL_ROUTES: dict[str, str] = {
    "qwen3_coder": "http://localhost:8001",
    "qwen38_briefs": "http://localhost:8002",
}

DEFAULT_MODEL = "qwen3_coder"


def log(msg: str) -> None:
    print(f"[proxy] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            models = []
            for name, url in MODEL_ROUTES.items():
                try:
                    with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as resp:
                        data = json.loads(resp.read())
                        models.extend(data.get("data", []))
                except Exception:
                    models.append({"id": name, "object": "model", "owned_by": "local"})
            self._json(200, {"object": "list", "data": models})
        elif self.path in ("/health", "/v1/health"):
            healthy = {}
            for name, url in MODEL_ROUTES.items():
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                        healthy[name] = resp.status == 200
                except Exception:
                    healthy[name] = False
            status = 200 if any(healthy.values()) else 503
            self._json(
                status, {"status": "ok" if status == 200 else "unhealthy", "models": healthy}
            )
        else:
            self._proxy_get()

    def do_POST(self) -> None:
        body = self._read_body()
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
        needs_think_strip = "/chat/completions" in self.path or "/responses" in self.path
        is_streaming = body.get("stream", False)
        log(f"POST {self.path} model={model} stream={is_streaming} strip_think={needs_think_strip}")
        if needs_think_strip and is_streaming:
            self._proxy_streaming_chat(backend, body)
        else:
            self._proxy_request(backend, body, strip_think=needs_think_strip)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _proxy_streaming_chat(self, backend: str, body: dict) -> None:
        parsed = urllib.parse.urlparse(backend)
        hostname = parsed.hostname
        port = parsed.port or 80
        if hostname is None:
            self._json(
                502, {"error": {"message": f"Backend error: invalid backend URL: {backend}"}}
            )
            return
        conn = http.client.HTTPConnection(hostname, port, timeout=300)
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            conn.request("POST", self.path, data, headers)
            resp = conn.getresponse()
        except Exception as e:
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

        in_think = False
        chunks_forwarded = 0
        chunks_suppressed = 0

        try:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                decoded = raw_line.decode("utf-8", errors="replace")
                stripped = decoded.strip()

                if not stripped:
                    self.wfile.write(b"\n")
                    self.wfile.flush()
                    continue

                if not stripped.startswith("data: "):
                    self.wfile.write(raw_line)
                    self.wfile.flush()
                    continue

                payload = stripped[6:]
                if payload == "[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    self.wfile.write(raw_line)
                    self.wfile.flush()
                    continue

                if "<think>" in payload:
                    log(
                        f"THINK detected in chunk keys={list(chunk.keys())} type={chunk.get('type', '?')}"
                    )
                if chunks_forwarded < 2:
                    log(f"FIRST CHUNK keys={list(chunk.keys())} type={chunk.get('type', '?')}")

                content, set_content = self._extract_content(chunk)

                if content is None:
                    if not in_think:
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        chunks_forwarded += 1
                    continue

                if not in_think and "<think>" in content:
                    in_think = True
                    before_think = content[: content.index("<think>")]
                    after_tag = content[content.index("<think>") + 7 :]
                    if before_think:
                        set_content(chunk, before_think)
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        chunks_forwarded += 1
                    if "</think>" in after_tag:
                        in_think = False
                        after_close = after_tag[after_tag.index("</think>") + 8 :].lstrip()
                        if after_close:
                            set_content(chunk, after_close)
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                            chunks_forwarded += 1
                    else:
                        chunks_suppressed += 1
                    continue

                if in_think:
                    if "</think>" in content:
                        in_think = False
                        after = content[content.index("</think>") + 8 :].lstrip()
                        if after:
                            set_content(chunk, after)
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                            chunks_forwarded += 1
                    else:
                        chunks_suppressed += 1
                    continue

                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                chunks_forwarded += 1

        except BrokenPipeError:
            log("Client disconnected (broken pipe)")
        finally:
            conn.close()
            log(f"Stream done: {chunks_forwarded} forwarded, {chunks_suppressed} think-suppressed")

    @staticmethod
    def _extract_content(chunk: dict) -> tuple:
        """Extract text content from either Chat Completions or Responses API format.
        Returns (content_str_or_None, setter_fn).
        """
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")

            def set_chat(c: dict, val: str) -> None:
                c["choices"][0]["delta"]["content"] = val

            return content, set_chat

        if chunk.get("type") == "response.output_text.delta":
            content = chunk.get("delta", "")

            def set_resp(c: dict, val: str) -> None:
                c["delta"] = val

            return content or None, set_resp

        if "delta" in chunk and isinstance(chunk["delta"], dict):
            content = chunk["delta"].get("text") or chunk["delta"].get("content")

            def set_delta(c: dict, val: str) -> None:
                if "text" in c["delta"]:
                    c["delta"]["text"] = val
                else:
                    c["delta"]["content"] = val

            return content, set_delta

        return None, lambda c, v: None

    @staticmethod
    def _strip_think_tags(resp_body: bytes) -> bytes:
        try:
            data = json.loads(resp_body)
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                content = msg.get("content")
                if content and "<think>" in content:
                    msg["content"] = THINK_RE.sub("", content).strip()
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        text = part.get("text", "")
                        if "<think>" in text:
                            part["text"] = THINK_RE.sub("", text).strip()
            return json.dumps(data).encode()
        except (json.JSONDecodeError, KeyError, TypeError):
            return resp_body

    def _proxy_request(self, backend: str, body: dict, strip_think: bool = False) -> None:
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
                if strip_think:
                    resp_body = self._strip_think_tags(resp_body)
                self.send_response(resp.status)
                for key in ("Content-Type",):
                    val = resp.headers.get(key)
                    if val:
                        self.send_header(key, val)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except BrokenPipeError:
            log("Client disconnected (broken pipe)")
        except Exception as e:
            try:
                if hasattr(e, "read"):
                    resp_body = e.read()  # type: ignore
                    code = getattr(e, "code", 502)
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_body)))
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
            except Exception:
                pass
            log(f"Backend error: {e}")
            try:
                self._json(502, {"error": {"message": f"Backend error: {e}"}})
            except BrokenPipeError:
                pass

    def _proxy_get(self) -> None:
        backend = list(MODEL_ROUTES.values())[0]
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
        except Exception as e:
            self._json(502, {"error": {"message": f"Backend error: {e}"}})

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
        pass


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    log(f"vLLM routing proxy on 0.0.0.0:{PORT}")
    log(f"Routes: {MODEL_ROUTES}")
    sys.stdout.flush()
    server = ThreadedServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
