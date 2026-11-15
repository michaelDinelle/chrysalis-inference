import { readFileSync } from "node:fs";

const HOUR = 3600;
const seconds = (s: string) => {
  const [h, m = "0", x = "0"] = s.replace(/^[+-]/, "").split(":");
  return (s.startsWith("-") ? -1 : 1) * (Number(h) * HOUR + Number(m) * 60 + Number(x));
};

function ruleDay(rule: string, year: number): number {
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0), jan1 = Date.UTC(year, 0, 1) / 1000;
  const m = /^M(\d+)\.(\d)\.(\d)$/.exec(rule);
  if (m) {
    const month = Number(m[1]) - 1, week = Number(m[2]), day = Number(m[3]), first = new Date(Date.UTC(year, month, 1)).getUTCDay();
    let date = 1 + ((day - first + 7) % 7) + (week - 1) * 7;
    const last = new Date(Date.UTC(year, month + 1, 0)).getUTCDate();
    while (date > last) date -= 7;
    return Date.UTC(year, month, date) / 1000;
  }
  if (rule.startsWith("J")) { const n = Number(rule.slice(1)); return jan1 + (n - 1 + (leap && n >= 60 ? 1 : 0)) * 86400; }
  return jan1 + Number(rule) * 86400;
}

export function posixOffset(tz: string, at: number): number | undefined {
  const name = "(<[^>]+>|[A-Za-z]{3,})", off = "([+-]?\\d{1,3}(?::\\d{1,2}){0,2})";
  const m = new RegExp(`^${name}${off}(?:${name}${off}?(?:,([^,/]+)(?:/${off})?,([^,/]+)(?:/${off})?)?)?$`).exec(tz);
  if (!m) return undefined;
  const std = -seconds(m[2]);
  if (!m[3]) return std;
  const dst = m[4] ? -seconds(m[4]) : std + HOUR, [startRule, startAt, endRule, endAt] = m[5] ? [m[5], m[6], m[7], m[8]] : ["M3.2.0", undefined, "M11.1.0", undefined];
  const year = new Date((at + std) * 1000).getUTCFullYear();
  const start = ruleDay(startRule, year) + (startAt ? seconds(startAt) : 2 * HOUR) - std, end = ruleDay(endRule, year) + (endAt ? seconds(endAt) : 2 * HOUR) - dst;
  const inside = start < end ? at >= start && at < end : !(at >= end && at < start);
  return inside ? dst : std;
}

export function tzifOffset(file: Uint8Array, at: number): number | undefined {
  const v = new DataView(file.buffer, file.byteOffset, file.byteLength);
  if (file.byteLength < 44 || String.fromCharCode(...file.subarray(0, 4)) !== "TZif") return undefined;
  const counts = (o: number) => [20, 24, 28, 32, 36, 40].map((k) => v.getUint32(o + k));
  let o = 0, wide = false;
  if (file[4] >= 0x32) {
    const [isut, isstd, leap, times, types, chars] = counts(0);
    o = 44 + times * 5 + types * 6 + chars + leap * 8 + isstd + isut;
    wide = true;
  }
  const [isut, isstd, leap, times, types, chars] = counts(o), size = wide ? 8 : 4, at0 = o + 44;
  const time = (i: number) => (wide ? Number(v.getBigInt64(at0 + i * 8)) : v.getInt32(at0 + i * 4));
  const kind = (i: number) => file[at0 + times * size + i];
  const utoff = (t: number) => v.getInt32(at0 + times * (size + 1) + t * 6);
  let last = -1;
  for (let i = 0; i < times; i++) if (time(i) <= at) last = i;
  const end = at0 + times * (size + 1) + types * 6 + chars + leap * (size + 4) + isstd + isut;
  const footer = wide ? new TextDecoder().decode(file.subarray(end)).split("\n")[1] ?? "" : "";
  if (footer && (times === 0 || last === times - 1)) {
    const fromRule = posixOffset(footer, at);
    if (fromRule !== undefined) return fromRule;
  }
  return types ? utoff(last >= 0 ? kind(last) : 0) : undefined;
}

export function utcOffset(at = Date.now(), env = process.env, read = (p: string) => readFileSync(p)): number {
  const tz = env.TZ?.replace(/^:/, ""), secs = Math.floor(at / 1000);
  const fromFile = (path: string) => { try { return tzifOffset(read(path), secs); } catch { return undefined; } };
  const found = tz ? (tz.startsWith("/") ? fromFile(tz) : fromFile(`/usr/share/zoneinfo/${tz}`) ?? posixOffset(tz, secs)) : fromFile("/etc/localtime");
  return found !== undefined ? found * 1000 : -new Date(at).getTimezoneOffset() * 60_000;
}
