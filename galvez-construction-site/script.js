/* ============================================================================
   GALVEZ CONSTRUCTION — interaction layer
   Principles: all animation via transform/opacity; scroll work is throttled
   through requestAnimationFrame; nothing that forces layout in the scroll path.
   ========================================================================== */
(() => {
  "use strict";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $  = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  /* ---------- 1. HERO REVEAL (clip-up mask + facade scale) ---------- */
  const hero = $("#hero");
  requestAnimationFrame(() => hero && hero.classList.add("is-lit"));

  /* ---------- 2. GENERIC SCROLL-IN REVEALS ---------- */
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); }
    }
  }, { threshold: 0.14, rootMargin: "0px 0px -8% 0px" });
  $$("[data-reveal],[data-tile]").forEach((el) => io.observe(el));

  /* ---------- 3. STAT COUNTERS ---------- */
  const fmt = (v, dec) => (dec ? v.toFixed(1) : Math.round(v).toLocaleString());
  const countIO = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target;
      const target = parseFloat(el.dataset.count);
      const suffix = el.dataset.suffix || "";
      const dec = String(el.dataset.count).includes(".");
      if (reduce || target === 0) { el.textContent = fmt(target, dec) + suffix; countIO.unobserve(el); continue; }
      const dur = 1400; const t0 = performance.now();
      const tick = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(target * eased, dec) + suffix;
        if (p < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      countIO.unobserve(el);
    }
  }, { threshold: 0.6 });
  $$(".hero__stats dt").forEach((el) => countIO.observe(el));

  /* ---------- 4. NAV: hide on scroll-down, condense on scroll ---------- */
  const nav = $("#nav");
  let lastY = window.scrollY, navTick = false;
  const onNav = () => {
    const y = window.scrollY;
    nav.classList.toggle("is-scrolled", y > 40);
    if (y > lastY && y > 320) nav.classList.add("is-hidden");
    else nav.classList.remove("is-hidden");
    lastY = y; navTick = false;
  };
  window.addEventListener("scroll", () => {
    if (!navTick) { requestAnimationFrame(onNav); navTick = true; }
  }, { passive: true });

  /* ---------- 5. SPECIALIZATION: sticky index swaps to active capability ---------- */
  const idxEl  = $("[data-index]");
  const wordEl = $("[data-word]");
  const descEl = $("[data-desc]");
  const progEl = $("[data-progress]");
  const caps   = $$("[data-cap]");
  let activeCap = -1;

  const setCap = (i) => {
    if (i === activeCap || !caps[i]) return;
    activeCap = i;
    const { title, desc } = caps[i].dataset;
    wordEl.classList.add("is-swap"); descEl.classList.add("is-swap");
    // swap content after the fade-out, then fade back in
    window.setTimeout(() => {
      idxEl.textContent  = String(i + 1).padStart(2, "0");
      wordEl.innerHTML   = title;
      descEl.textContent = desc;
      wordEl.classList.remove("is-swap"); descEl.classList.remove("is-swap");
    }, reduce ? 0 : 180);
    if (progEl) progEl.style.width = ((i + 1) / caps.length * 100) + "%";
  };

  const capIO = new IntersectionObserver((entries) => {
    // choose the entry most centered in the viewport
    let best = null;
    for (const e of entries) if (e.isIntersecting) {
      if (!best || e.intersectionRatio > best.intersectionRatio) best = e;
    }
    if (best) setCap(caps.indexOf(best.target));
  }, { threshold: [0.35, 0.6, 0.85], rootMargin: "-25% 0px -25% 0px" });
  caps.forEach((c) => capIO.observe(c));
  setCap(0);

  /* ---------- 6. PARALLAX (transform only, rAF-batched) ---------- */
  const layers = $$("[data-parallax]");
  let pTick = false;
  const onParallax = () => {
    const vh = window.innerHeight;
    for (const el of layers) {
      const r = el.getBoundingClientRect();
      const mid = r.top + r.height / 2;
      const off = (mid - vh / 2) / vh;            // -1..1 across viewport
      const k = parseFloat(el.dataset.parallax) || 0;
      el.style.transform = `translate3d(0, ${(off * k * 100).toFixed(2)}px, 0)`;
    }
    pTick = false;
  };
  if (!reduce && layers.length) {
    window.addEventListener("scroll", () => {
      if (!pTick) { requestAnimationFrame(onParallax); pTick = true; }
    }, { passive: true });
    onParallax();
  }

  /* ---------- 7. ETHOS: word-by-word structural reveal ---------- */
  const split = $("[data-split]");
  if (split) {
    const words = split.textContent.trim().split(/\s+/);
    split.innerHTML = words.map((w) => `<span class="word">${w}</span>`).join(" ");
    const spans = $$(".word", split);
    let last = -1;
    const revealTo = (n) => {
      if (n === last) return; last = n;
      spans.forEach((s, i) => s.classList.toggle("is-on", i <= n));
    };
    const ethosIO = new IntersectionObserver((es) => {
      es.forEach((e) => { if (e.isIntersecting) startEthos(); else stopEthos(); });
    }, { threshold: 0 });
    let raf = null;
    const startEthos = () => { if (raf || reduce) { revealTo(spans.length); return; } tickEthos(); };
    const stopEthos  = () => { if (raf) cancelAnimationFrame(raf), raf = null; };
    const tickEthos = () => {
      const r = split.getBoundingClientRect();
      const vh = window.innerHeight;
      // progress: 0 when quote enters lower third, 1 when it passes upper third
      const p = Math.min(1, Math.max(0, (vh * 0.85 - r.top) / (r.height + vh * 0.35)));
      revealTo(Math.floor(p * (spans.length - 1)));
      raf = requestAnimationFrame(tickEthos);
    };
    ethosIO.observe(split);
  }

  /* ---------- 8. BLUEPRINT FORM ---------- */
  const form = $("#intakeForm");
  if (form) {
    const ok = $("#formOk");
    const openedAt = performance.now(); // for the bot timing check below
    const mark = (field, bad) => field.closest(".field").classList.toggle("invalid", bad);
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();

      // --- anti-bot: honeypot filled or superhuman submit speed → drop silently.
      // (First line of defense only; the backend must repeat these checks.)
      const trap = $("#company_website");
      if ((trap && trap.value !== "") || performance.now() - openedAt < 2000) {
        ok.hidden = false; // pretend success so bots learn nothing
        return;
      }

      const name = $("#name"), email = $("#email"), scope = $("#scope");
      const emailOk = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value);
      let bad = false;
      [[name, !name.value.trim()], [email, !emailOk], [scope, !scope.value]]
        .forEach(([f, isBad]) => { mark(f, isBad); if (isBad) bad = true; });
      if (bad) { form.querySelector(".field.invalid input,.field.invalid select")?.focus(); return; }
      ok.hidden = false;
      form.querySelectorAll("input,select,textarea,button").forEach((n) => (n.disabled = true));
      // In production this posts to a backend / form service over HTTPS.
      // Values are never interpolated into the DOM, so there is no client-side
      // XSS sink here — but the backend must still validate and escape.
      // console.log("intake", Object.fromEntries(new FormData(form)));
    });
    // clear invalid state as the user corrects a field
    form.addEventListener("input", (e) => {
      const f = e.target.closest(".field"); if (f) f.classList.remove("invalid");
    });
  }

  /* ---------- 9. footer year ---------- */
  const yr = $("#year"); if (yr) yr.textContent = "2026";
})();
