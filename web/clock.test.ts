import { expect, test } from "bun:test";
import { utcOffset } from "./clock";

const ZONES = ["America/Edmonton", "Europe/London", "Europe/Dublin", "Australia/Sydney", "Australia/Lord_Howe", "America/St_Johns", "Asia/Kolkata",
  "Pacific/Chatham", "America/Santiago", "America/Nuuk", "Antarctica/Troll", "Africa/Casablanca", "America/Phoenix", "UTC", "Pacific/Kiritimati"];

test("the terminal's clock reads the system's own zone data, as date does, where the runtime's copy is behind: summer and winter, half and quarter hours, southern and inverted summers, past the file's last transition and before its first", () => {
  const times = [Date.UTC(1950, 5, 1), Date.UTC(2015, 0, 15), Date.UTC(2026, 2, 8, 9, 30), Date.UTC(2026, 6, 1), Date.UTC(2026, 9, 4, 15, 59), Date.UTC(2026, 10, 1, 7, 59),
    Date.UTC(2026, 10, 1, 8, 1), Date.UTC(2031, 3, 6, 2, 30), Date.UTC(2038, 6, 1), Date.UTC(2045, 11, 31, 23)];
  const ours: string[] = [], dates: string[] = [];
  for (const zone of ZONES) for (const at of times) {
    const z = Bun.spawnSync(["date", "-d", `@${at / 1000}`, "+%z"], { env: { TZ: zone }, stdout: "pipe" }).stdout.toString().trim();
    const ms = utcOffset(at, { TZ: zone }), mins = Math.abs(ms / 60_000);
    ours.push(`${zone} ${at} ${ms < 0 ? "-" : "+"}${String(Math.floor(mins / 60)).padStart(2, "0")}${String(mins % 60).padStart(2, "0")}`);
    dates.push(`${zone} ${at} ${z === "-0000" ? "+0000" : z}`);
  }
  expect(ours).toEqual(dates);
  expect(new Set(dates.map((d) => d.split(" ")[2])).size).toBeGreaterThan(12);
});
