export type Drop = { kind: "pin"; index: number } | { kind: "unpin" } | { kind: "folder"; who: string } | { kind: "settle" } | { kind: "active" };
interface Options { verb: (row: HTMLElement, drop: Drop) => string | null; done: (row: HTMLElement, drop: Drop) => void; started: () => void; ended: () => void }

const DISTANCE = 6, EDGE = 10;

export function sidebarDrag(list: HTMLElement, o: Options) {
  list.addEventListener("pointerdown", (down) => {
    const picked = (down.target as Element).closest<HTMLElement>("a.conv[data-key]");
    if (!picked || !down.isPrimary || down.button !== 0 || down.pointerType === "touch" || (down.target as Element).closest("button, input, [data-card]")) return;
    let row = picked;
    const wasPinned = !!row.closest(".folder.pinned"), wasSettled = !!row.closest(".folder.settled");
    let dragging = false, drop: Drop | undefined, done = false, badge: HTMLElement | undefined;
    let scrolled = 0, top = 0, height = 0;
    let pins: { top: number; bottom: number; mid: number }[] = [], pinned: { top: number; bottom: number } | undefined, shelf: { top: number } | undefined;
    let folders: { who: string; top: number; bottom: number }[] = [];
    const made: HTMLElement[] = [];
    let pinStrip: HTMLElement | undefined, settleStrip: HTMLElement | undefined, bar: HTMLElement | undefined;
    const suppress = (e: Event) => { e.stopPropagation(); e.preventDefault(); document.removeEventListener("click", suppress, true); };
    const clearSelection = () => document.getSelection()?.removeAllRanges();
    const strip = (label: string, over: DOMRect) => {
      const el = document.createElement("div");
      el.className = "drag-strip";
      el.textContent = label;
      el.style.cssText = `left:${over.left}px;width:${over.width}px;top:${over.top}px;height:${over.height}px`;
      document.body.append(el);
      made.push(el);
      return el;
    };
    const start = () => {
      if (!row.isConnected) {
        const again = [...list.querySelectorAll<HTMLElement>("a.conv[data-key]")].find((el) => el.dataset.key === row.dataset.key && !!el.closest(".folder.pinned") === wasPinned);
        if (!again) return false;
        row = again;
      }
      dragging = true;
      o.started();
      const now = row.getBoundingClientRect(), box = list.getBoundingClientRect(), rect = (el: Element | null) => el?.getBoundingClientRect();
      scrolled = list.scrollTop;
      top = now.top;
      height = now.height;
      pins = [...list.querySelectorAll<HTMLElement>(".folder.pinned a.conv")].filter((el) => el !== row).map((el) => { const r = el.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, mid: r.top + r.height / 2 }; });
      pinned = rect(list.querySelector(".folder.pinned"));
      shelf = rect(list.querySelector(".folder.settled"));
      folders = [...list.querySelectorAll<HTMLElement>(".folder[data-who]")].map((el) => { const r = el.getBoundingClientRect(); return { who: el.dataset.who!, top: r.top, bottom: r.bottom }; });
      const above = list.previousElementSibling?.getBoundingClientRect(), below = list.nextElementSibling?.getBoundingClientRect();
      if (!pinned && above) pinStrip = strip("Pinned", above);
      if (!shelf && below) settleStrip = strip("Settled", below);
      bar = document.createElement("div");
      bar.className = "drag-bar";
      bar.hidden = true;
      bar.style.left = `${box.left + 8}px`;
      bar.style.width = `${box.width - 16}px`;
      document.body.append(bar);
      made.push(bar);
      list.classList.add("sorting");
      row.classList.add("dragging");
      badge = document.createElement("span");
      badge.className = "conv-verb";
      badge.setAttribute("role", "status");
      badge.hidden = true;
      (row.querySelector(".conv-slot") ?? row).append(badge);
      document.addEventListener("click", suppress, true);
      document.addEventListener("selectionchange", clearSelection);
      clearSelection();
      return true;
    };
    const aim = (seen: number, scroll: number, pointer: number): Drop | undefined => {
      const y = seen + scroll, box = list.getBoundingClientRect(), tries: Drop[] = [];
      if (pinned ? y <= pinned.bottom : pointer <= box.top + EDGE) tries.push({ kind: "pin", index: pins.filter((p) => p.mid < y).length });
      else if (shelf ? y >= shelf.top : pointer >= box.bottom - EDGE) tries.push({ kind: "settle" });
      else {
        const folder = folders.find((f) => y >= f.top && y <= f.bottom);
        if (folder) tries.push({ kind: "folder", who: folder.who });
        if (wasPinned) tries.push({ kind: "unpin" });
        if (wasSettled) tries.push({ kind: "active" });
      }
      return tries.find((t) => o.verb(row, t) !== null);
    };
    const paint = (scroll: number) => {
      for (const el of list.querySelectorAll(".drop")) el.classList.remove("drop");
      list.querySelector(".pin-head")?.classList.toggle("target", drop?.kind === "pin");
      pinStrip?.classList.toggle("target", drop?.kind === "pin");
      settleStrip?.classList.toggle("target", drop?.kind === "settle");
      if (drop?.kind === "folder") list.querySelector(`.folder[data-who="${CSS.escape(drop.who)}"]`)?.classList.add("drop");
      if (drop?.kind === "settle") list.querySelector(".folder.settled")?.classList.add("drop");
      const at = drop?.kind === "pin" && pins.length ? (drop.index < pins.length ? pins[drop.index].top - 2 : pins[pins.length - 1].bottom + 1) - scroll : undefined, box = list.getBoundingClientRect();
      if (bar) {
        bar.hidden = at === undefined || at < box.top || at > box.bottom;
        if (at !== undefined) bar.style.top = `${at}px`;
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
      const box = list.getBoundingClientRect();
      if (e.clientY < box.top + 28) list.scrollTop -= 10;
      else if (e.clientY > box.bottom - 28) list.scrollTop += 10;
      const scroll = list.scrollTop - scrolled, at = Math.max(box.top, Math.min(top + (e.clientY - down.clientY), box.bottom - height));
      row.style.transform = `translateY(${at - (top - scroll)}px)`;
      drop = aim(at + height / 2, scroll, e.clientY);
      paint(scroll);
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
      for (const el of made) el.remove();
      list.classList.remove("sorting");
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
