import { expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { cardPng, castPrompt, readCast, type Card } from "./cast";

const PY = `${import.meta.dir}/../data/venv/bin/python`;
const py = (code: string, ...args: string[]) => {
  const r = Bun.spawnSync([PY, "-I", "-c", code, ...args]);
  if (r.exitCode !== 0) throw new Error(r.stderr.toString());
  return r.stdout.toString();
};
const card: Card = { spec: "chara_card_v3", spec_version: "3.0", data: { name: "Mirela", description: "{{char}} keeps bees. Ünïcödé ✓ 🐝",
  first_mes: "Hello, {{user}}.", alternate_greetings: ["Oh. You again."], tags: ["bees"], extensions: { other_app: { kept: true } },
  character_book: { entries: [{ keys: ["hive"], content: "The hive is old." }], extensions: {} } } };

test("a card exported as a PNG is the picture as it was, with the card in Pillow's eyes in both chunks; a card Pillow writes as SillyTavern does is read", async () => {
  const dir = mkdtempSync(`${tmpdir()}/chrysalis-cards-`);
  try {
    py(`import sys, base64, json\nfrom PIL import Image, PngImagePlugin\nim = Image.new("RGB", (48, 64), (200, 30, 90))\n` +
       `info = PngImagePlugin.PngInfo()\ninfo.add_text("chara", base64.b64encode(json.dumps({"spec": "chara_card_v2", "data": {"name": "Old", "description": "x"}}).encode()).decode())\n` +
       `im.save(sys.argv[1], pnginfo=info)`, `${dir}/drawn.png`);
    const out = await cardPng(card, new Blob([readFileSync(`${dir}/drawn.png`)], { type: "image/png" }));
    await Bun.write(`${dir}/out.png`, out);
    const seen = JSON.parse(py(`import sys, base64, json\nfrom PIL import Image\nim = Image.open(sys.argv[1]); im.load()\n` +
      `print(json.dumps({"size": im.size, "pixel": im.convert("RGB").getpixel((10, 10)), "keys": sorted(im.text), "charas": open(sys.argv[1], "rb").read().count(b"tEXtchara\\0"),\n` +
      `"v3": json.loads(base64.b64decode(im.text["ccv3"])), "v2": json.loads(base64.b64decode(im.text["chara"]))}))`, `${dir}/out.png`));
    expect(seen.size).toEqual([48, 64]);
    expect(seen.pixel).toEqual([200, 30, 90]);
    expect([seen.keys, seen.charas]).toEqual([["ccv3", "chara"], 1]);
    expect([seen.v3.spec, seen.v3.spec_version, seen.v2.spec, seen.v2.spec_version]).toEqual(["chara_card_v3", "3.0", "chara_card_v2", "2.0"]);
    expect(seen.v3.data).toEqual({ group_only_greetings: [], ...card.data });
    expect(seen.v2.data).toEqual(card.data);

    const back = await readCast(new File([readFileSync(`${dir}/out.png`)], "out.png"));
    expect(back && "card" in back && back.card.data).toEqual(seen.v3.data);
    const v2only = await readCast(new File([readFileSync(`${dir}/drawn.png`)], "drawn.png"));
    expect(v2only && "card" in v2only && [v2only.card.spec, v2only.card.data.name]).toEqual(["chara_card_v2", "Old"]);
    py(`import sys\nfrom PIL import Image\nImage.new("RGB", (4, 4)).save(sys.argv[1])`, `${dir}/plain.png`);
    expect(await readCast(new File([readFileSync(`${dir}/plain.png`)], "plain.png"))).toBeNull();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("JSON cards of each version and the group file are read, a card's personality and example lines as part of its description; other JSON is not a card", async () => {
  const read = (j: unknown, name = "x.json") => readCast(new File([JSON.stringify(j)], name));
  const v1 = await read({ name: "Ion", description: "A ferryman.", first_mes: "Hi." });
  expect(v1 && "card" in v1 && [v1.card.spec, v1.card.data.name, v1.card.data.first_mes]).toEqual(["chara_card_v2", "Ion", "Hi."]);
  const v2 = await read({ spec: "chara_card_v2", spec_version: "2.0", data: { name: "Ana", description: "{{char}} bakes.", personality: "kind", mes_example: "<START>\n{{user}}: Bread?\n{{char}}: Rye.",
    post_history_instructions: "Stay in character.", creator: "someone" } });
  expect(v2 && "card" in v2 && v2.card.data).toEqual({ name: "Ana", description: "{{char}} bakes.\n\n{{char}}'s personality: kind\n\nHow {{char}} talks:\n{{user}}: Bread?\n{{char}}: Rye.",
    post_history_instructions: "Stay in character.", creator: "someone" });
  expect(v2 && "card" in v2 && castPrompt("", [v2.card.data], { name: "Robin" })).toBe("Ana bakes.\n\nAna's personality: kind\n\nHow Ana talks:\nRobin: Bread?\nAna: Rye.");
  const g = await read({ spec: "chrysalis_group", name: "Court", members: ["Ana", "Ion"], scenario: "A feast." });
  expect(g).toEqual({ group: { name: "Court", members: ["Ana", "Ion"], cards: [], scenario: "A feast." } });
  expect(await read({ name: "package", version: "1.0.0" })).toBeNull();
  expect(await read([1, 2, 3])).toBeNull();
});
