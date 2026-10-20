# Implementation Plan — image generation fidelity (text, background, artwork)

**Status:** planned, nothing implemented yet.

**Scope:** the three issues filed after the review of the 9 latest generated images
(production, `designs/2026/10/...`):

| Issue | Title | Branch |
|-------|-------|--------|
| [#187](https://github.com/idsysapps/store-listing-generator/issues/187) | briefs render wrong text and get dropped - composite the headline and tagline deterministically again | `feat/deterministic-text-compositing` |
| [#188](https://github.com/idsysapps/store-listing-generator/issues/188) | the printed background is not white and the print file can be a product mockup | `fix/prompt-background-and-artwork` |
| [#189](https://github.com/idsysapps/store-listing-generator/issues/189) | the OCR text gate compares the whole image as one blob and passes altered text | `fix/ocr-gate-line-matching` |

**Related:** [#133](https://github.com/idsysapps/store-listing-generator/issues/133) (integrated
text via AnyText / TextDiffuser-2 on the DGX) is the long-term answer for the flat look of the
compositor. Its body is stale: it assumes the Pillow compositor is still in the pipeline, but it
was deleted in `e825b0e` (#111).

---

## 1. What the review found

Nine briefs with images, checked against the brief row in Postgres:

| brief | brief says | the image shows |
|---|---|---|
| 89 sticker_vinyl `text_top_bottom` | `🎃` / `Spooky Vibes Only` | `#。OOOOOOOOO` + a stray heart; the tagline **twice** |
| 88 dtf_apparel `text_top` | `Witch Please` / `Spell-check required`, `#000000` | **"Witch Pleas"**, tagline **missing**, text orange not black |
| 86 dtf_apparel `text_top` | `Bag of Spooky Fun` / `Holds all your Halloween loot`, `#000000` | **"Bag of... Spooky Fun."**, tagline **missing**, text yellow not black |
| 84 dtf_apparel `text_top` | `Spooktacular Match` / `Perfect for haunted nights` | **"Spooktacular Match\nMatch"**, tagline **missing**, plus a hallucinated "Trick OR teet" sign |
| 79 sublimation `text_top_bottom` | `Bone Appétit` / `Pumpkin Spice Latte` | "Bone Appétit."; tagline **twice**, one copy garbled ("Bemen' Sppice Latte") |
| 82 dtf_apparel `text_top` | `Trick or Treat`, `#000000` | text correct, but rendered **white** |
| 80 dtf_apparel `text_top` | `Pumpkin Everything` | text correct, but the print file is a **photo of a hoodie** |
| 78 sublimation `text_top` | `Fixer Upper: Ghosts Included` | text correct, plus a hallucinated "FOR SALE" sign |
| 87 sublimation `full_bleed` | skeleton at a desk, white paper background | good art, but a **solid black background** |

Headline wrong in 3/9, tagline missing in 4/9, duplicated in 2/9, requested colour ignored in 3/9,
hallucinated extra text in 4/9, wrong artefact type in 1/9, wrong background in 1/9.

## 2. Root causes

1. **Flux is asked to spell.** `build_flux_prompt` puts the headline and the tagline into the
   diffusion prompt. Flux Schnell's exact-match accuracy is ~14% (#133). The colour in the prompt
   is ignored too. This is the cause of every text defect above.
2. **The prompt structure lets the brief override the product.** The white-background requirement
   for `sublimation` / `sticker_vinyl` loses against a "high-contrast black" visual style
   (brief 87), and nothing tells the LLM that a scene is *print artwork* rather than a product
   photo (brief 80).
3. **The gate does not measure the image.** It matches the expected text against the whole
   transcription with a fuzzy 0.85 threshold, and the transcription itself can be wrong (brief 88:
   "Witch Pleas" on the canvas, "Witch Please\nWitch\nWitch\nPlease" reported).

## 3. Order of work

```
#187  deterministic text compositing   ──┐
                                          ├──► #189  strict, line-based OCR gate
#188  prompt: background + artwork      ──┘
```

- **#187 and #188 both touch `build_flux_prompt`.** Do the prompt work in one pass so the two do
  not fight over the same function - #188's changes land with #187's.
- **#189 is only meaningful once the text is ours.** With compositing, the gate asserts our own
  text and can be exact; before that it would fight the model.

## 4. Todo

### #187 - deterministic text compositing

- [ ] restore `text_compositor.py` from `e825b0e^` (`src/store_listing/orchestration/text_compositor.py`,
      93 lines) and its test file, as a starting point - not as-is
- [ ] vendor an OFL font under `src/store_listing/assets/fonts/` (the runtime image has **no
      fonts**: `/usr/share/fonts` is empty and Pillow 12 does not bundle DejaVu;
      `ImageFont.load_default(size=...)` works - Pillow ships Aileron - but it is not a display face
      and has no emoji)
- [ ] strip the headline/tagline from the Flux prompt for every layout; ask for artwork with empty
      space where the text goes ("no lettering, no text, no words")
- [ ] composite the headline (top zone) and tagline (bottom zone) per layout, in the brief's
      `font_color`, with the existing near-white-on-white correction
- [ ] composite **before** the background removal for `dtf_apparel`, so the print file keeps the text
- [ ] decide what happens to an emoji headline (brief 89): a font with the glyph, or drop it from
      the brief
- [ ] tests: `tests/unit/test_text_compositor.py` (layout zones, colour, wrapping, missing glyph),
      `tests/unit/test_image_pipeline.py` (prompt carries no text, composited image carries the
      exact text), `tests/unit/test_image_generation.py`
- [ ] PR: `feat/deterministic-text-compositing` (closes #187)

### #188 - background and artwork fidelity

- [ ] put the background requirement last in the prompt, so a visual style cannot override it
- [ ] tell the brief generator that the scene is **flat print artwork**, never a garment, mug, tote
      or blank with the design on it
- [ ] add "no product mockup, no garment, no photo of a printed item" to the `dtf_apparel` product
      context
- [ ] tests: `tests/unit/test_image_generation.py` (background instruction is last),
      `tests/unit/test_design_briefs.py` (the brief prompt asks for print artwork)
- [ ] PR: `fix/prompt-background-and-artwork` (closes #188) - merge with #187

### #189 - the OCR gate measures the image

- [ ] compare line by line: each expected text has to be found in a single line of the
      transcription
- [ ] exact match after normalisation for composited text; keep the tolerant path only for
      model-drawn text
- [ ] report per-text verdicts, so a missing tagline and a garbled headline are distinguishable
- [ ] log the transcription verbatim when nothing matches
- [ ] tests: `tests/unit/test_text_validation.py`
- [ ] PR: `fix/ocr-gate-line-matching` (closes #189)

### After the three

- [ ] re-generate the reviewed briefs (86, 88, 89, 79, 84, 80) and re-run the review
- [ ] the 54 upsert events and the 72 stuck material events are independent of this work
- [ ] #133 stays open: when AnyText / TextDiffuser-2 is on the DGX, the compositor can be swapped
      for the model and the flat look goes away

## 5. Risks and notes

- **The compositor is flat.** That is why it was deleted in #111. "Exact text" beats "integrated
  but wrong text" for a print product, and #133 is the path back to integrated text.
- **The dtf_apparel order matters.** Compositing before the background removal keeps the text in
  the transparent print file; compositing after would leave the raw preview without it.
- **Deploy ordering.** The app image carries the Python changes, so the app pods have to roll
  (`helm upgrade` from the published chart, or `kubectl rollout restart`). The `dtf_apparel`
  background removal and the text both happen in the worker, so the two app containers (worker and
  beat) have to be on the same version.
- **The material pipeline is unaffected** by all of this - it is the trend-service (Java) that
  consumes `materials:events`, the image pipeline is the Python app.
