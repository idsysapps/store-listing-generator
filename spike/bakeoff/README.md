# Pipeline bake-off

A timeboxed spike. It answers one question: which rendering architecture should
be the default, now that the current pipeline has three text paths and two image
models but still ships text that disappears on the shirt, chroma-key residue and
sticker fills?

This directory is not production code. It does not write to the production
database or S3.

## Eval set

`briefs.jsonl` is a snapshot of 14 completed DTF briefs:

| Briefs | What they cover |
|---|---|
| 205, 206, 207, 209, 210, 211, 213 | Known failures: text contrast, text placement, chroma residue, sticker fill |
| 204, 208, 212 | Hero and no-text baseline |
| 196, 198, 199, 201 | Neon, typographic, sticker and chalk methods |

It covers all three voices, both garment colours, `caption`/`type_hero`/
`no_text`, and every text method in use.

## Arms

- **current** - the production render already stored for the brief. It is the
  baseline, not a re-render.
- **hybrid** - Flux draws the isolated subject on a magenta key; BiRefNet keys
  it; the deterministic compositor draws the words. No text plan carrier, no
  model-drawn text, no embedding profile, and no dark-garment background
  instruction.
- **text_first** - Qwen-Image draws the scene and the headline. The descriptor
  is composited afterwards. This is the current hero/model-text strategy.

The hybrid arm deliberately leaves the accent word off. Brief 213 showed that
`loud` was drawing the first word in a dark accent on a Jet Black shirt. A
type_hero brief with no `scene_description` (204) falls back to a text-only
render instead of failing the arm.

## Run

The harness has to run where ComfyUI is reachable. In this cluster that is the
app pod, which can reach `http://192.168.2.123:8188`:

```bash
kubectl -n store-generator exec deploy/store-listing-store-listing-app \
  -c celery-worker -- python /tmp/spike/bakeoff/run_spike.py \
  --briefs /tmp/spike/bakeoff/briefs.jsonl \
  --out /tmp/spike/out \
  --comfy-url http://192.168.2.123:8188 \
  --arms hybrid,text_first \
  --seeds 2 \
  --text-first-seeds 1
```

Then copy the output back and score it:

```bash
python spike/bakeoff/score_spike.py \
  --briefs spike/bakeoff/briefs.jsonl \
  --manifest spike/out/manifest.jsonl \
  --out spike/out \
  --ocr-arms current,text_first

python spike/bakeoff/make_contact_sheets.py \
  --briefs spike/bakeoff/briefs.jsonl \
  --manifest spike/out/manifest.jsonl \
  --out spike/out
```

The current renders live in `spike/out/current/`. They are downloaded from the
running app and are gitignored with the rest of `spike/out/`.

## Metrics

- `coverage` - opaque pixels / canvas. A DTF print should not be a solid sheet.
- `magenta_pct` - magenta pixels / ink. Chroma-key residue.
- `solid_light_pct` - a large light fill inside the artwork bounding box. This
  catches the cream sticker fill that coverage alone misses.
- `near_white_border` - a near-white ring around the artwork bounding box.
- `text_overlap` - share of the composited text that sits on the subject.
- `contrast` - WCAG contrast of the planned text ink against the recommended
  garment colour.
- `accent_contrast` - the same for the voice's accent word, when one is drawn.
- `ocr_pass` - vision OCR on the current and text-first arms.

The hybrid text is drawn from a font, so its spelling is exact by construction;
OCR is not needed for that arm.

## Results

The write-up, tables and contact sheets are in
[`docs/spikes/2026-11-pipeline-bakeoff.md`](../../docs/spikes/2026-11-pipeline-bakeoff.md).

## Scoring tests

```bash
pytest tests/unit/test_spike_scoring.py
```
