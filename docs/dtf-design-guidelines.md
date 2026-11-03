# DTF design guidelines: the garment is the background

A DTF print is ink on the shirt. Every transparent pixel in the print file is
unprinted, so the garment colour shows through. That makes transparency a
design material, not just a cut-out.

Large solid fields of ink make a shirt feel stiff and plastic. Keep the printed
area to the idea: the subject, the words, and any carrier the design actually
needs.

## Rules

- **Do not print a backdrop.** No background rectangle, no full-bleed scene,
  no solid plate behind the type. The scene is the subject, not a scene with a
  background.
- **Design the negative space.** Let the shirt colour show around and through
  the design. A shape can be defined by what is *not* printed.
- **Keep the ink to the subject, the words and intentional carriers.** A sign,
  label, headstone, banner or screen is a carrier: the words belong on it. A
  plate behind a slogan is not.
- **Prefer type directly on the garment.** Use a solid band or bar only when
  the text plan names a carrier.
- **No default die-cut sticker border.** A white outline around the artwork is
  printed ink. Use it only when the design needs a contrast outline, and keep
  it thin.
- **Design for dark and light shirts.** Do not rely on a printed white
  background for legibility. Use contrast, a thin outline or a halo sparingly.
- **Stay inside the placement.** No edge-to-edge art unless the product is
  full-bleed (for example sublimation, where the background is printed).

## What this looks like in the pipeline

- For `dtf_apparel`, the artwork is generated on a magenta key background and
  keyed out. The magenta is **not** printed; the alpha channel is the print
  area.
- The keyed matte gets a sub-pixel alpha feather before the despill and the
  colour bleed, so a subject's edge does not print as a staircase and the
  colour under the soft edge is cleaned up afterwards. Letter counters are
  punched out after the feather, so a hole stays a hole.
- The compositor reads the keyed alpha and moves a caption to a clear band
  above or below the art when the subject reaches into the template's clear
  zone. A clear template zone is kept as it is; if the ink leaves no clear
  band, the template zone is the fallback.
- `build_flux_prompt` asks for an isolated subject with no printed background
  or backdrop.
- The brief prompt carries `DTF_RULES`: the garment is the background, keep the
  ink to the subject/words/carriers, and use negative space.
- The text compositor only draws a solid bar when the text plan needs a
  carrier. A typographic plan puts the words straight on the shirt.
- The hero/model-text prompts use a screen-printed apparel graphic for DTF, not
  a die-cut sticker, so no default white border is drawn.
- Every DTF render logs its opaque coverage (`alpha > 0` as a share of the
  canvas). Coverage above 65% logs a warning: the design may print as a large
  solid area. This is a warning, not a rejection; a large subject can be
  intentional.
- With `IMAGE_CANDIDATE_COUNT=2`, the renderer compares two candidates and
  keeps the one with less type on the subject, less key residue, less light
  fill and a cleaner edge. The QA gate measures text contrast against the
  recommended garment, not against the magenta key.
- `TEXT_EMBED_PROFILE` turns on DTF-safe text embedding for planned carriers:
  `off` is the default, `v11`/`v12` are the standard wear levels, and `v13`
  is the heavy distressed level. The font still owns spelling. Glyph pixels
  stay at 100% opacity; texture is applied to the shading layer, and distress
  is a binary knockout, never a semi-transparent letter.

## Garment material and colour

The brief recommends materials from the published catalogue. Those rows come
from the cost-book material stream and are validated against what cost-book
actually offers; never invent a garment colour.

- The **first recommended material is the garment**. Its `color` is the shirt
  colour, and the mockup/listing uses that colour.
- **Black or dark garment** (Jet Black, Dark Heather): the black areas of the
  design are the shirt, not ink. Draw light highlights and lines, drop
  near-black pixels from the print file, and use a light ink for the text.
- **Light or white garment**: the design is printed in ink. Keep the black
  areas and use a dark ink for the text.
- A design built for a dark garment is not offered on a light one unless the
  black is printed; the recommended material decides.

## Checks

- Review the design on a dark and a light shirt mockup, not only on white.
- Look for a printed rectangle, a solid band behind the text, or a sticker
  border that was not intentional.
- Check the opaque coverage in the render log. A design that covers most of
  its placement should have a reason.
