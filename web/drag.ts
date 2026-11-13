export type Drop = { kind: "pin"; index: number } | { kind: "unpin" } | { kind: "folder"; who: string } | { kind: "settle" } | { kind: "active" };
interface Options { verb: (row: HTMLElement, drop: Drop) => string | null; done: (row: HTMLElement, drop: Drop) => void; started: () => void; ended: () => void }
interface Parts { pins: HTMLElement; list: HTMLElement; shelf: HTMLElement }

const DISTANCE = 6;

export function sidebarDrag(root: HTMLElement, at: Parts, o: Options) {
  root.addEventListener("pointerdown", (down) => {
    const picked = (down.target as Element).closest<HTMLElement>("a.conv[data-key]");
    if (!picked || !down.isPrimary || down.button !== 0 || down.pointerType === "touch" || (down.target as Element).closest("button, input, [data-card]")) return;
    let row = picked;
    const wasPinned = at.pins.contains(row), wasSettled = at.shelf.contains(row);
    let dragging = false, drop: Drop | undefined, done = false, ghost: HTMLElement | undefined, badge: HTMLElement | undefined, bar: HTMLElement | undefined;
    let scrolled = 0, top = 0, height = 0;
    let pins: { top: number; bottom: number; mid: number }[] = [], folders: { who: string; top: number; bottom: number }[] = [];
    const suppress = (e: Event) => { e.stopPropagation(); e.preventDefault(); document.removeEventListener("click", suppress, true); };
    const clearSelection = () => document.getSelection()?.removeAllRanges();
    const start = () => {
      if (!row.isConnected) {
        const again = [...root.querySelectorAll<HTMLElement>("a.conv[data-key]")].find((el) => el.dataset.key === row.dataset.key && at.pins.contains(el) === wasPinned);
        if (!again) return false;
        row = again;
      }
      dragging = true;
      o.started();
      const now = row.getBoundingClientRect(), box = root.getBoundingClientRect();
      scrolled = at.list.scrollTop;
      top = now.top;
      height = now.height;
      pins = [...at.pins.querySelectorAll<HTMLElement>("a.conv")].filter((el) => el !== row).map((el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, mid: r.top + r.height / 2 }; });
      folders = [...at.list.querySelectorAll<HTMLElement>(".folder[data-who]")].map((el) => { const r = el.getBoundingClientRect(); return { who: el.dataset.who!, top: r.top, bottom: r.bottom }; });
      ghost = row.cloneNode(true) as HTMLElement;
      ghost.removeAttribute("id");
      ghost.classList.add("drag-ghost");
      ghost.style.cssText = `left:${now.left}px;top:${now.top}px;width:${now.width}px;height:${now.height}px`;
      badge = document.createElement("span");
      badge.className = "conv-verb";
      badge.setAttribute("role", "status");
      badge.hidden = true;
      (ghost.querySelector(".conv-slot") ?? ghost).append(badge);
      bar = document.createElement("div");
      bar.className = "drag-bar";
      bar.hidden = true;
      bar.style.left = `${box.left + 8}px`;
      bar.style.width = `${box.width - 16}px`;
      document.body.append(ghost, bar);
      row.classList.add("lifted");
      root.classList.add("sorting");
      document.addEventListener("click", suppress, true);
      document.addEventListener("selectionchange", clearSelection);
      clearSelection();
      return true;
    };
    const aim = (y: number): Drop | undefined => {
      const above = at.pins.getBoundingClientRect(), below = at.shelf.getBoundingClientRect(), tries: Drop[] = [];
      if (y <= above.bottom) tries.push({ kind: "pin", index: pins.filter((p) => p.mid < y).length });
      else if (y >= below.top) tries.push({ kind: "settle" });
      else {
        const inList = y + at.list.scrollTop - scrolled, folder = folders.find((f) => inList >= f.top && inList <= f.bottom);
        if (folder) tries.push({ kind: "folder", who: folder.who });
        if (wasPinned) tries.push({ kind: "unpin" });
        if (wasSettled) tries.push({ kind: "active" });
      }
      return tries.find((t) => o.verb(row, t) !== null);
    };
    const paint = () => {
      for (const el of at.list.querySelectorAll(".drop")) el.classList.remove("drop");
      at.pins.classList.toggle("target", drop?.kind === "pin");
      at.shelf.classList.toggle("target", drop?.kind === "settle");
      if (drop?.kind === "folder") at.list.querySelector(`.folder[data-who="${CSS.escape(drop.who)}"]`)?.classList.add("drop");
      const line = drop?.kind === "pin" && pins.length ? (drop.index < pins.length ? pins[drop.index].top - 2 : pins[pins.length - 1].bottom + 1) : undefined;
      if (bar) {
        bar.hidden = line === undefined;
        if (line !== undefined) bar.style.top = `${line}px`;
      }
      const verb = drop ? o.verb(row, drop) : null;
      if (badge) { badge.hidden = !verb; badge.textContent = verb ?? ""; }
    };
    const move = (e: PointerEvent) => {
      if (done || e.pointerId !== down.pointerId) return;
      if ((e.buttons & 1) === 0) return finish(!dragging);
      if (!dragging) {
        if (Math.hypot(e.clientX - down.clientX, e.clientY - down.clientY) <= DISTANCE) return;
        if (!start()) return finish(true);
      }
      if (e.cancelable) e.preventDefault();
      const box = root.getBoundingClientRect(), scroll = at.list.getBoundingClientRect();
      if (e.clientY > scroll.top && e.clientY < scroll.top + 28) at.list.scrollTop -= 10;
      else if (e.clientY < scroll.bottom && e.clientY > scroll.bottom - 28) at.list.scrollTop += 10;
      const label = at.pins.querySelector(".pin-head")?.getBoundingClientRect().bottom ?? box.top, floor = at.shelf.querySelector(".shelf-head, .settled-head")?.getBoundingClientRect().top ?? box.bottom;
      const y = Math.max(label, Math.min(top + (e.clientY - down.clientY), floor - height));
      ghost!.style.top = `${y}px`;
      drop = aim(Math.max(box.top + 1, Math.min(e.clientY, box.bottom - 1)));
      paint();
    };
    const up = (e: PointerEvent) => { if (e.pointerId === down.pointerId) finish(false); };
    const cancel = () => finish(true);
    const key = (e: KeyboardEvent) => { if (e.code === "Escape") cancel(); };
    const stop = (e: Event) => e.preventDefault();
    function finish(cancelled: boolean) {
      if (done) return;
      done = true;
      document.removeEventListener("pointermove", move, true);
      document.removeEventListener("pointerup", up, true);
      document.removeEventListener("pointercancel", cancel, true);
      document.removeEventListener("keydown", key, true);
      document.removeEventListener("dragstart", stop);
      document.removeEventListener("selectionchange", clearSelection);
      if (!dragging) return;
      setTimeout(() => document.removeEventListener("click", suppress, true));
      ghost?.remove();
      bar?.remove();
      for (const el of [root, at.pins, at.shelf]) el.classList.remove("sorting", "target");
      const to = cancelled ? undefined : drop;
      if (to) o.done(row, to);
      o.ended();
    }
    document.addEventListener("pointermove", move, { capture: true, passive: false });
    document.addEventListener("pointerup", up, true);
    document.addEventListener("pointercancel", cancel, true);
    document.addEventListener("keydown", key, true);
    document.addEventListener("dragstart", stop);
  });
}
