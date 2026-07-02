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

| File                 | Role                                                            |
|----------------------|-----------------------------------------------------------------|
| `index.html`         | Semantic markup for all sections + inline SVG concrete textures |
| `styles.css`         | The full design system, motion tokens, and responsive rules     |
| `script.js`          | Scroll reveals, sticky capability swap, counters, parallax, form |
| `assets/fonts/`      | Self-hosted variable fonts (Space Grotesk + Inter, ~70 KB total) |
| `_headers`           | Security headers for Netlify / Cloudflare Pages                 |
| `vercel.json`        | Security headers for Vercel                                     |
| `nginx.conf.example` | Hardened self-hosted nginx config (TLS, headers, method filter) |
| `.well-known/security.txt` | RFC 9116 vulnerability-disclosure contact                 |

## Security architecture

No site is "unhackable," but this one keeps its attack surface close to the
theoretical minimum for a website: it is fully static (no server-side code, no
database, no sessions, no cookies — nothing to inject into or exfiltrate from)
and hardened in depth:

- **Zero external origins.** Fonts are self-hosted; there are no CDNs, trackers,
  analytics, or third-party scripts. Nothing to supply-chain-attack, nothing to
  MITM, no data leaves the page.
- **Strict Content-Security-Policy** — `default-src 'none'` with narrow `'self'`
  allowances only. No inline scripts, no `eval`, no external loads. A meta-tag
  CSP ships in `index.html` as a fallback; the real policy (plus
  `frame-ancestors 'none'`) is delivered via HTTP headers from whichever config
  matches your host (`_headers`, `vercel.json`, or `nginx.conf.example`).
- **Full header suite** on deploy: HSTS (2 years, preload), `X-Frame-Options:
  DENY` (no clickjacking), `X-Content-Type-Options: nosniff`, `Referrer-Policy`,
  `Permissions-Policy` (camera/mic/geolocation all denied), COOP + CORP.
- **No client-side XSS sinks.** Form values are never interpolated into the
  DOM or URLs; the only `innerHTML` writes use static, author-controlled
  strings.
- **Form anti-abuse**: a honeypot field plus a submit-timing check drop naive
  bots client-side (with a fake success so they learn nothing), and inputs are
  length-capped. Web3Forms adds its own server-side spam filtering on top.
- **`security.txt`** so researchers know where to report issues.

## Making the form deliver (2 minutes)

The form works out of the box: with no configuration it opens the visitor's
email client pre-filled with the intake details (addressed to
`FALLBACK_MAILTO` in `script.js` — change it to the real inbox).

For proper in-page delivery, wire up Web3Forms (free, no account):

1. Go to <https://web3forms.com>, enter the inbox that should receive
   intakes, and copy the access key they email you.
2. Paste it into `WEB3FORMS_ACCESS_KEY` at the top of the form section in
   `script.js`.
3. Done — submissions now arrive by email with spam filtering, and the CSP
   (`connect-src`) already allows exactly `api.web3forms.com` and nothing else.

The access key is designed to be public (it only lets people send *you*
email), so committing it is fine.

## Publishing (≈$10/yr total)

1. Put this folder in its own GitHub repo.
2. Cloudflare Pages → connect the repo → framework preset *None*, output `/`.
   The `_headers` file ships the full security-header suite automatically.
3. Buy the domain at Cloudflare Registrar or Porkbun, attach it in Pages,
   enable **DNSSEC** and **Always Use HTTPS**.
4. Turn on 2FA for GitHub, Cloudflare, and the registrar.
5. Sanity-check at securityheaders.com (expect an A+).

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

Space Grotesk (display) + Inter (body), self-hosted as variable-font woff2
files in `assets/fonts/` — no third-party font CDN at runtime. System-font
fallbacks are declared if the files somehow fail to load.
