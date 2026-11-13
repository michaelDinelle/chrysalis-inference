(() => {
  const S = (window.__chrysalis ??= { mode: "use", pin: null, prev: null, hover: null, code: document.currentScript?.textContent ?? "" });
  const send = (m) => parent.postMessage(m, "*");
  const store = () => {
    const kept = new Map();
    const api = { getItem: (k) => (kept.has(String(k)) ? kept.get(String(k)) : null), setItem: (k, v) => void kept.set(String(k), String(v)),
      removeItem: (k) => void kept.delete(String(k)), clear: () => kept.clear(), key: (i) => [...kept.keys()][i] ?? null };
    return new Proxy(api, {
      get: (t, k) => (k === "length" ? kept.size : k in t ? t[k] : typeof k === "string" && kept.has(k) ? kept.get(k) : undefined),
      set: (t, k, v) => (t.setItem(k, v), true),
      deleteProperty: (t, k) => (t.removeItem(k), true),
      has: (t, k) => k in t || kept.has(String(k)),
      ownKeys: () => [...kept.keys()],
      getOwnPropertyDescriptor: (t, k) => (kept.has(String(k)) ? { value: kept.get(String(k)), enumerable: true, configurable: true, writable: true } : undefined),
    });
  };
  S.stores ??= { localStorage: store(), sessionStorage: store() };
  for (const name of Object.keys(S.stores)) Object.defineProperty(window, name, { value: S.stores[name], configurable: true });
  S.jar ??= new Map();
  Object.defineProperty(document, "cookie", { configurable: true, get: () => [...S.jar].map(([k, v]) => `${k}=${v}`).join("; "),
    set: (line) => {
      const [pair, ...rest] = String(line).split(";"), at = pair.indexOf("="), name = (at < 0 ? "" : pair.slice(0, at)).trim(), value = pair.slice(at + 1).trim();
      const gone = rest.some((r) => /^\s*max-age\s*=\s*(0|-\d+)\s*$/i.test(r) || (/^\s*expires\s*=/i.test(r) && Date.parse(r.slice(r.indexOf("=") + 1)) < Date.now()));
      if (gone) S.jar.delete(name);
      else S.jar.set(name, value);
    } });
  const fail = (err, line, col) => {
    const text = String(err ?? "").replace(/^Uncaught /, "").replace(/^(\w+): Failed to execute 'write' on 'Document': /, "$1: ");
    const found = !text.startsWith("SyntaxError") || !line ? [] : [...new Set([...(S.prev ?? "").matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)]
      .map((m) => m[1].split("\n")[line - 1] ?? "").filter((l) => l.trim() && l.length >= col - 1).map((l) => l.trim()))];
    const at = found.length === 1 ? found[0] : "";
    if (text) send({ chrysalisFail: text, at: at.length > 240 ? `${at.slice(0, 240)}…` : at });
  };
  addEventListener("error", (e) => fail(e.message, e.lineno, e.colno));
  addEventListener("unhandledrejection", (e) => fail(e.reason?.stack ? `${e.reason.name}: ${e.reason.message}` : e.reason, 0, 0));
  const html = document.documentElement;

  const windowSized = (el) => {
    const r = el.getBoundingClientRect();
    return r.width >= innerWidth * 0.9 && r.height >= innerHeight * 0.9;
  };
  const fills = () => [...document.querySelectorAll("canvas")].some(windowSized) ||
    [...(document.body?.children ?? [])].flatMap((c) => [c, ...c.children]).some((el) => /^(fixed|absolute)$/.test(getComputedStyle(el).position) && windowSized(el));
  let sent = "", later = 0;
  const post = (loaded) => {
    const m = { chrysalisHeight: Math.ceil(html.getBoundingClientRect().height), fills: fills(), view: innerHeight, ...(loaded && { loaded: true }) };
    const key = JSON.stringify({ ...m, wide: innerWidth });
    if (key !== sent || loaded) send(m);
    sent = key;
  };
  new ResizeObserver(() => post()).observe(html);
  new MutationObserver(() => { later ||= requestAnimationFrame(() => { later = 0; post(); }); }).observe(html, { childList: true, subtree: true });
  addEventListener("load", () => setTimeout(() => post(S.written)));

  const ui = (tag, css) => {
    const el = document.createElement(tag);
    el.setAttribute("data-chrysalis-ui", "");
    if (tag === "style") el.textContent = css; else el.style.cssText = css;
    return el;
  };
  const outline = (strong) => ui("div", "position:fixed;pointer-events:none;z-index:2147483647;display:none;border-radius:3px;box-sizing:border-box;" +
    `border:${strong ? 2 : 1.5}px ${strong ? "solid" : "dashed"} #e24d85;box-shadow:0 0 0 ${strong ? 4 : 3}px rgb(226 77 133 / ${strong ? 0.28 : 0.16})`);
  const hoverBox = outline(false), pinBox = outline(true), cursor = ui("style", "");
  const CURSOR = { use: "", select: "*{cursor:default!important}", bucket: "*{cursor:cell!important}", text: "*{cursor:text!important}" };
  const place = (box, cx) => {
    const el = cx === null ? null : document.querySelector(`[data-cx="${cx}"]`), r = el && rectOf(el);
    if (!r) return void (box.style.display = "none");
    if (!box.isConnected) html.append(box);
    Object.assign(box.style, { display: "block", left: `${r.left - 3}px`, top: `${r.top - 3}px`, width: `${r.width + 6}px`, height: `${r.height + 6}px` });
  };
  const sceneOf = () => (window.AFRAME && document.querySelector("a-scene")?.camera ? document.querySelector("a-scene") : null);
  function entityAt(e) {
    const scene = sceneOf();
    if (!scene || !(e.target instanceof HTMLCanvasElement)) return null;
    const T = window.AFRAME.THREE, r = scene.canvas.getBoundingClientRect(), ray = new T.Raycaster();
    ray.setFromCamera(new T.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), scene.camera);
    for (const hit of ray.intersectObject(scene.object3D, true)) {
      let seen = true;
      for (let o = hit.object; o; o = o.parent) seen &&= o.visible;
      if (!seen) continue;
      for (let o = hit.object; o; o = o.parent) {
        const el = o.el?.closest?.("[data-cx]");
        if (el && el !== scene) return el;
      }
    }
    return null;
  }
  function rectOf(el) {
    const scene = sceneOf();
    if (!scene || !el.object3D || el === scene || !scene.contains(el)) return el.getBoundingClientRect();
    const T = window.AFRAME.THREE, box = new T.Box3().setFromObject(el.object3D), c = scene.canvas.getBoundingClientRect();
    if (box.isEmpty()) return null;
    const xs = [], ys = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const p = new T.Vector3(x, y, z).project(scene.camera);
      if (p.z > 1) continue;
      xs.push(c.left + ((p.x + 1) / 2) * c.width);
      ys.push(c.top + ((1 - p.y) / 2) * c.height);
    }
    if (!xs.length) return null;
    const left = Math.max(c.left, Math.min(...xs)), top = Math.max(c.top, Math.min(...ys));
    return { left, top, width: Math.min(c.right, Math.max(...xs)) - left, height: Math.min(c.bottom, Math.max(...ys)) - top };
  }
  const show = () => {
    if (!cursor.isConnected) html.append(cursor);
    if (cursor.textContent !== CURSOR[S.mode]) cursor.textContent = CURSOR[S.mode];
    place(hoverBox, S.mode === "use" || editing ? null : S.hover);
    place(pinBox, S.mode === "select" ? S.pin : null);
  };
  addEventListener("scroll", show, { passive: true, capture: true });
  addEventListener("resize", show);
  const follow = () => { if (S.mode !== "use" && sceneOf()) { show(); requestAnimationFrame(follow); } };

  const partAt = (e) => entityAt(e) ?? (e.target instanceof Element ? e.target.closest("[data-cx]") : null);
  const cxOf = (el) => (el ? Number(el.getAttribute("data-cx")) : null);
  const retypable = (el) => !el.matches("script,style,textarea,input,select,svg,svg *,canvas,iframe,video,audio,img,html,head") &&
    ![...el.querySelectorAll("*")].some((n) => !n.hasAttribute("data-cx") || n.matches("script,style,svg,canvas,iframe,video,audio"));
  let editing = null, before = "";
  const finish = (keep) => {
    const el = editing;
    if (!el) return;
    editing = null;
    el.removeAttribute("contenteditable");
    if (!keep) el.innerHTML = before;
    else if (el.innerHTML !== before) {
      const copy = el.cloneNode(true);
      copy.querySelectorAll("[data-cx]").forEach((n) => n.removeAttribute("data-cx"));
      send({ chrysalisText: { cx: cxOf(el), html: copy.innerHTML } });
    }
    show();
  };
  const retype = (el) => {
    if (!retypable(el)) return send({ chrysalisNote: "That text is made by the page's own code: ask for it to be changed" });
    editing = el;
    before = el.innerHTML;
    el.setAttribute("contenteditable", "true");
    el.focus();
    show();
  };

  addEventListener("pointermove", (e) => {
    if (S.mode === "use" || editing) return;
    const cx = cxOf(partAt(e));
    if (cx === S.hover) return;
    S.hover = cx;
    send({ chrysalisHover: cx });
    show();
  }, true);
  html.addEventListener("pointerleave", () => {
    if (S.hover === null) return;
    S.hover = null;
    send({ chrysalisHover: null });
    show();
  });
  const held = (e) => S.mode !== "use" && !(editing && editing.contains(e.target));
  for (const type of ["pointerdown", "mousedown", "pointerup", "mouseup", "dblclick", "auxclick", "contextmenu", "submit"])
    addEventListener(type, (e) => { if (held(e)) { e.preventDefault(); e.stopImmediatePropagation(); } }, true);
  addEventListener("click", (e) => {
    if (!held(e)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    finish(true);
    const el = partAt(e), cx = cxOf(el);
    if (S.mode === "select") {
      S.pin = cx === S.pin ? null : cx;
      send({ chrysalisPick: S.pin });
    } else if (el && S.mode === "bucket") send({ chrysalisPaint: cx });
    else if (el && S.mode === "text") retype(el);
    show();
  }, true);
  addEventListener("keydown", (e) => {
    if (editing) {
      if (e.key === "Escape") finish(false);
      else if (e.key === "Enter" && !e.shiftKey) finish(true);
      else return;
    } else {
      const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
      const act = mod && k === "z" ? "undo" : mod && k === "d" ? "duplicate" : S.mode === "select" && S.pin !== null && /^(delete|backspace)$/.test(k) ? "delete"
        : k === "escape" && S.pin !== null ? "unpin" : null;
      if (S.mode === "use" || !act) return;
      send({ chrysalisKey: act });
    }
    e.preventDefault();
    e.stopImmediatePropagation();
  }, true);
  addEventListener("focusout", (e) => { if (editing && e.target === editing) finish(true); }, true);

  const managed = (el) => [...el.childNodes].filter((n) => n.nodeType === 3 || (n.nodeType === 1 && n.hasAttribute("data-cx")));
  const HELD = "text/chrysalis-held";
  const hold = (node) => {
    if (node.nodeType === 1) for (const s of [node, ...node.querySelectorAll("script")]) if (s.tagName === "SCRIPT") s.type = HELD;
    return node;
  };
  const same = (a, b) => a.nodeType === b.nodeType && (a.nodeType === 3 || a.tagName === b.tagName);
  function syncEl(live, next, prev, attrs = true) {
    if (next.tagName === "SCRIPT" || prev?.isEqualNode(next)) return;
    if (attrs) {
      for (const a of next.attributes) if (prev?.getAttribute(a.name) !== a.value && live.getAttribute(a.name) !== a.value) live.setAttribute(a.name, a.value);
      if (prev) for (const a of prev.attributes) if (!next.hasAttribute(a.name)) live.removeAttribute(a.name);
    }
    if (/^(STYLE|TITLE|TEXTAREA)$/.test(next.tagName)) {
      if (prev?.textContent !== next.textContent && live.textContent !== next.textContent) live.textContent = next.textContent;
      return;
    }
    const have = managed(live), want = managed(next), was = prev ? managed(prev) : [];
    const ids = new Set(want.map((w) => w.id).filter(Boolean)), byId = new Map(have.filter((h) => h.id).map((h) => [h.id, h]));
    const wasId = new Map(was.filter((p) => p.id).map((p) => [p.id, p]));
    const loose = have.filter((h) => !(h.id && ids.has(h.id))), wasLoose = was.filter((p) => !(p.id && ids.has(p.id)));
    const used = new Set();
    let last = null, k = 0;
    for (const w of want) {
      let h = w.id ? byId.get(w.id) : undefined, p = w.id ? wasId.get(w.id) : undefined;
      if (!h) { h = loose[k]; p = wasLoose[k]; k++; }
      let node;
      if (h && !used.has(h) && same(h, w)) {
        if (w.nodeType === 3) { if (!(p?.nodeType === 3 && p.data === w.data) && h.data !== w.data) h.data = w.data; }
        else syncEl(h, w, p && same(p, w) ? p : null);
        node = h;
        if (last && last.compareDocumentPosition(h) & Node.DOCUMENT_POSITION_PRECEDING) last.after(h);
      } else {
        node = hold(document.importNode(w, true));
        if (h && !used.has(h)) h.replaceWith(node); else if (last) last.after(node); else live.prepend(node);
        if (node.nodeType === 1) node.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 240, easing: "ease-out" });
      }
      used.add(h);
      used.add(node);
      last = node;
    }
    for (const h of have) if (!used.has(h)) h.remove();
  }
  function sync(src) {
    const parse = (s) => new DOMParser().parseFromString(s, "text/html");
    const next = parse(src), prev = S.prev === null ? null : parse(S.prev);
    for (const k of ["head", "body"]) syncEl(document[k], next[k], prev?.[k] ?? null, next[k].hasAttribute("data-cx"));
    S.prev = src;
  }
  function withLibs(page, libs) {
    if (!libs) return { page, map: "" };
    const blobs = (S.blobs ??= new Map()), names = {};
    const url = (code) => blobs.get(code) ?? blobs.set(code, URL.createObjectURL(new Blob([code], { type: "text/javascript" }))).get(code);
    for (const [spec, code] of Object.entries(libs)) {
      if (/^(https?:)?\/\//.test(spec)) page = page.split(spec).join(url(code));
      else names[spec] = url(code);
    }
    page = page.replace(/(<script\b[^>]*?)\sintegrity\s*=\s*("[^"]*"|'[^']*')/gi, "$1");
    if (!Object.keys(names).length) return { page, map: "" };
    let merged = false;
    page = page.replace(/(<script\b[^>]*type\s*=\s*["']importmap["'][^>]*>)([\s\S]*?)(<\/script>)/i, (all, open, json, close) => {
      try {
        const map = JSON.parse(json);
        map.imports = { ...map.imports, ...names };
        merged = true;
        return open + JSON.stringify(map) + close;
      } catch { return all; }
    });
    return { page, map: merged ? "" : `<script type="importmap" data-chrysalis-ui>${JSON.stringify({ imports: names })}<\/script>` };
  }
  function write(src, libs) {
    S.prev = src;
    S.hover = null;
    S.written = true;
    const { page: body, map } = withLibs(src, libs);
    const page = body.replace(/^(\s*(?:<!--[\s\S]*?-->\s*)*<!doctype[^>]*>)?/i, (d) => `${d}<script data-chrysalis-ui>${S.code}<\/script>${map}`);
    document.open();
    try {
      document.write(page);
    } catch (err) {
      fail(`${err.name}: ${err.message}`, 0, 0);
    }
    document.close();
  }

  addEventListener("message", (e) => {
    const m = e.source === parent && e.data?.chrysalis;
    if (!m) return;
    if (m.mode && m.mode !== S.mode) {
      finish(true);
      S.mode = m.mode;
      S.hover = null;
      follow();
    }
    if ("pin" in m) S.pin = m.pin;
    if (typeof m.page === "string" && m.write) return write(m.page, m.libs);
    if (typeof m.page === "string") sync(m.page);
    show();
  });
  if (!S.ready) {
    S.ready = true;
    send({ chrysalisReady: true });
  }
  show();
})();
