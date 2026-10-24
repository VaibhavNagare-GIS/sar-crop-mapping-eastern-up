const root = document.documentElement;
const loaderBar = document.querySelector(".loader__bar");
const imgs = [...document.querySelectorAll(".hero img")];
let done = 0;
const bump = () => loaderBar && loaderBar.style.setProperty("--p", Math.min(1, ++done / (imgs.length + 1)));
const ready = Promise.all([
  document.fonts.ready.then(bump),
  ...imgs.map((i) => i.decode().catch(() => {}).then(bump)),
]);
const minShow = new Promise((r) => setTimeout(r, Math.max(0, 1100 - performance.now())));
const cap = new Promise((r) => setTimeout(r, 5000));
Promise.race([Promise.all([ready, minShow]), cap]).then(() => {
  root.classList.add("is-ready");
  document.querySelector(".loader")?.setAttribute("inert", "");
});

const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const nav = document.querySelector(".nav");
const setSolid = (y) => nav.classList.toggle("is-solid", y > 40);
let lenis = null;

if (!reduce && window.Lenis) {
  lenis = new Lenis({ lerp: 0.1 });
  lenis.on("scroll", (e) => setSolid(e.scroll));
  if (window.gsap && window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((t) => lenis.raf(t * 1000));
    gsap.ticker.lagSmoothing(0);
  } else {
    const raf = (t) => { lenis.raf(t); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
  }
} else {
  addEventListener("scroll", () => setSolid(scrollY), { passive: true });
}
setSolid(scrollY);

document.addEventListener("click", (e) => {
  const a = e.target.closest('a[href^="#"]');
  if (!a || a.getAttribute("href") === "#") return;
  const target = document.querySelector(a.getAttribute("href"));
  if (!target) return;
  e.preventDefault();
  if (lenis) lenis.scrollTo(target, { offset: -nav.offsetHeight });
  else target.scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
});

if (!reduce && window.gsap && window.ScrollTrigger) {
  const hero = document.querySelector(".hero");
  const tl = gsap.timeline({
    scrollTrigger: { trigger: hero, start: "top top", end: "bottom top", scrub: true, invalidateOnRefresh: true },
  });
  const h = () => hero.offsetHeight;
  tl.to(".hero__sky", { y: () => h() * 0.16, ease: "none" }, 0)
    .to(".hero__sat", { y: () => h() * 0.05, ease: "none" }, 0)
    .to(".hero__fg", { y: () => -h() * 0.05, ease: "none" }, 0);
  document.fonts.ready.then(() => ScrollTrigger.refresh());
}

import("./map.js");
