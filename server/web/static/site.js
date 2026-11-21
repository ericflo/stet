// Stet: the little the pages need. No framework, no build step.
(function () {
  "use strict";
  var doc = document, root = doc.documentElement;

  // --- theme ---
  doc.addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-theme-toggle]");
    if (!b) return;
    var dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("stet-theme", root.dataset.theme); } catch (e) {}
  });

  // --- copy buttons ---
  doc.addEventListener("click", function (ev) {
    var b = ev.target.closest("[data-copy]");
    if (!b) return;
    var box = b.closest(".cmd");
    var src = box && box.querySelector("[data-copy-text]");
    var text = src ? src.textContent : (box ? box.querySelector("pre").textContent : "");
    var done = function () { var was = b.textContent; b.textContent = "copied"; setTimeout(function () { b.textContent = was; }, 1400); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text.trim()).then(done, function () {});
    } else {
      var ta = doc.createElement("textarea");
      ta.value = text.trim(); doc.body.appendChild(ta); ta.select();
      try { doc.execCommand("copy"); done(); } catch (e) {}
      ta.remove();
    }
  });

  // --- tabs (install: sh / PowerShell) ---
  function showTab(group, name) {
    group.querySelectorAll("[data-tab]").forEach(function (t) { t.setAttribute("aria-selected", String(t.dataset.tab === name)); });
    group.querySelectorAll("[data-pane]").forEach(function (p) { p.hidden = p.dataset.pane !== name; });
  }
  doc.addEventListener("click", function (ev) {
    var t = ev.target.closest("[data-tab]");
    if (t) showTab(t.closest("[data-tabs]"), t.dataset.tab);
  });
  if (/Win/.test(navigator.platform || "")) doc.querySelectorAll("[data-tabs]").forEach(function (g) { showTab(g, "ps"); });

  // --- countdowns ---
  function span(s) {
    if (s < 90) return s + "s";
    if (s < 5400) return Math.round(s / 60) + "m";
    if (s < 129600) { var h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60); return m ? h + "h " + m + "m" : h + "h"; }
    return Math.round(s / 86400) + "d";
  }
  function tick() {
    var now = Date.now() / 1000;
    doc.querySelectorAll("[data-until]").forEach(function (el) {
      var d = Math.round(Number(el.dataset.until) - now);
      el.textContent = d > 0 ? "in " + span(d) : "any moment now";
    });
  }
  tick();
  setInterval(tick, 1000);

  // --- the galley proof ---
  var galley = doc.querySelector("[data-galley]");
  if (galley) {
    var play = function () { galley.classList.remove("play"); void galley.offsetWidth; galley.classList.add("play"); };
    play();
    var again = galley.querySelector("[data-replay]");
    if (again) again.addEventListener("click", play);
  }

  // --- the round diagram: its pencil strokes are drawn when it comes into view ---
  var diagrams = doc.querySelectorAll("[data-play]");
  if (diagrams.length && "IntersectionObserver" in window) {
    var seen = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("play"); seen.unobserve(e.target); }
      });
    }, { threshold: 0.5 });
    diagrams.forEach(function (d) { seen.observe(d); });
  }

  // --- a row that opens: open the one a link points at ---
  function openTarget() {
    if (!location.hash) return;
    var el = doc.getElementById(location.hash.slice(1));
    var d = el && (el.matches("details") ? el : el.querySelector("details"));
    if (d) d.open = true;
  }
  openTarget();
  window.addEventListener("hashchange", openTarget);

  // --- live parts: re-read the page and swap in the parts marked data-live ---
  var live = doc.querySelectorAll("[data-live][id]");
  if (live.length) {
    setInterval(function () {
      if (doc.hidden) return;
      fetch(location.href, { headers: { "X-Stet-Live": "1" } }).then(function (r) { return r.ok ? r.text() : null; }).then(function (html) {
        if (!html) return;
        var next = new DOMParser().parseFromString(html, "text/html");
        live.forEach(function (el) {
          var fresh = next.getElementById(el.id);
          if (!fresh || fresh.innerHTML === el.innerHTML) return;
          // What the reader has opened stays open.
          var open = {};
          el.querySelectorAll("details[open]").forEach(function (d) { var k = d.id || (d.parentElement && d.parentElement.id); if (k) open[k] = true; });
          el.innerHTML = fresh.innerHTML;
          el.querySelectorAll("details").forEach(function (d) { var k = d.id || (d.parentElement && d.parentElement.id); if (k && open[k]) d.open = true; });
        });
        tick();
      }).catch(function () {});
    }, 30000);
  }

  // --- chart tooltips, with a crosshair on line charts ---
  var tip = doc.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "status");
  doc.body.appendChild(tip);
  var NS = "http://www.w3.org/2000/svg", marks = [];
  function clear() {
    tip.classList.remove("on");
    marks.forEach(function (m) { m.remove(); });
    marks = [];
  }
  function show(el, ev) {
    clear();
    tip.textContent = el.dataset.tip;
    var r = el.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top;
    var svg = el.ownerSVGElement;
    if (svg && el.dataset.x) {
      var vb = svg.viewBox.baseVal, sr = svg.getBoundingClientRect(), k = sr.width / vb.width;
      var cx = Number(el.dataset.x), cy = Number(el.dataset.y);
      var line = doc.createElementNS(NS, "line");
      line.setAttribute("class", "cross");
      line.setAttribute("x1", cx); line.setAttribute("x2", cx);
      line.setAttribute("y1", el.getAttribute("y")); line.setAttribute("y2", Number(el.getAttribute("y")) + Number(el.getAttribute("height")));
      var dot = doc.createElementNS(NS, "circle");
      dot.setAttribute("class", "focus"); dot.setAttribute("cx", cx); dot.setAttribute("cy", cy); dot.setAttribute("r", 4.5);
      svg.appendChild(line); svg.appendChild(dot);
      marks = [line, dot];
      x = sr.left + cx * k; y = sr.top + cy * k;
    } else if (ev && r.height > 60) {
      y = ev.clientY;
    }
    var half = tip.offsetWidth / 2 + 8;
    tip.style.left = Math.max(half, Math.min(window.innerWidth - half, x)) + "px";
    tip.style.top = Math.max(40, y) + "px";
    tip.classList.add("on");
  }
  doc.addEventListener("pointermove", function (ev) {
    var el = ev.target.closest && ev.target.closest("[data-tip]");
    if (el) show(el, ev); else if (tip.classList.contains("on")) clear();
  });
  doc.addEventListener("pointerleave", clear);
  window.addEventListener("scroll", clear, { passive: true });
})();

// A strip of rounds to pick from shows the picked one, however far along it is.
(function () {
  document.querySelectorAll(".pick").forEach(function (p) {
    var on = p.querySelector(".on");
    if (on && p.scrollWidth > p.clientWidth) p.scrollLeft = on.offsetLeft - p.offsetLeft - (p.clientWidth - on.offsetWidth) / 2;
  });
})();
