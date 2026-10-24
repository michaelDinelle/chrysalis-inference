export type Theme = "system" | "light" | "dark";
const system = matchMedia("(prefers-color-scheme: dark)");
const changed = new Set<() => void>();

export function chosen(): Theme {
  try { return (localStorage.getItem("theme") as Theme | null) ?? "system"; } catch { return "system"; }
}
export const isDark = () => document.documentElement.dataset.theme === "dark";
export function onTheme(f: () => void) { changed.add(f); }
export function choose(t: Theme) {
  try { localStorage.setItem("theme", t); } catch {  }
  apply(t);
}
function apply(t = chosen()) {
  const look = t === "system" ? (system.matches ? "dark" : "light") : t;
  if (document.documentElement.dataset.theme === look) return;
  document.documentElement.dataset.theme = look;
  for (const f of changed) f();
}
system.addEventListener("change", () => apply());
apply();
