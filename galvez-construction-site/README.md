# Galvez Construction — Marketing Site

A premium, single-page marketing website for **Galvez Construction LLC**, a
high-end construction company specializing in architectural and structural
concrete.

**Design language:** *Industrial Editorial Architecture* — heavy, precise,
monolithic. Soft concrete grey (`#F5F5F7`), charcoal steel (`#111111`), and
international orange (`#FF5A36`) used very sparingly as a structural accent.

## Run it

No build step, no dependencies. Open the file directly:

```bash
open index.html          # macOS
xdg-open index.html      # Linux
```

Or serve locally (recommended so fonts/relative paths behave):

```bash
python3 -m http.server 8080
# → http://localhost:8080
```

## Structure

| File         | Role                                                            |
|--------------|-----------------------------------------------------------------|
| `index.html` | Semantic markup for all sections + inline SVG concrete textures |
| `styles.css` | The full design system, motion tokens, and responsive rules     |
| `script.js`  | Scroll reveals, sticky capability swap, counters, parallax, form |

## Sections

1. **Hero** — clip-up masked headline reveal, monolithic SVG facade that scales
   1.12 → 1.0 on load, animated stat counters, running capability marquee.
2. **Specialization** — sticky oversized index on the left that swaps content as
   the capability panels scroll past on the right (42/58 asymmetric split).
3. **Ethos** — a dark editorial break with a word-by-word scroll-linked reveal.
4. **Proof** — an intentionally uneven precision grid; project captions wipe in
   on a clip-mask hover.
5. **Contact** — a "blueprint" intake form with client-side validation and a
   reactive-scale CTA.

## Motion notes

- Core easing everywhere: `cubic-bezier(.16, 1, .3, 1)`.
- All scroll work is throttled through `requestAnimationFrame`; animations touch
  only `transform` / `opacity` to hold 60fps.
- Fully honors `prefers-reduced-motion: reduce`.

## Swapping in real photography

The gallery and hero currently use procedurally generated concrete textures
(CSS gradients + an SVG grain filter) so the site is 100% self-contained. To use
real photos, replace the `background` on each `.tile__img` / `.frame__media` with
your image, e.g.:

```css
.tile--1 .tile__img { background: url("assets/meridian.jpg") center/cover; }
```

## Fonts

Space Grotesk (display) + Inter (body) via Google Fonts, with system-font
fallbacks if the CDN is unavailable.
