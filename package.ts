import { $ } from "bun";
import { chmodSync, closeSync, copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, openSync, readdirSync, readFileSync, readSync, renameSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { deflateSync } from "node:zlib";

const ROOT = import.meta.dir, OS = process.platform, WIN = OS === "win32", MAC = OS === "darwin";
const arg = (name: string) => { const i = process.argv.indexOf(`--${name}`); return i > 0 ? process.argv[i + 1] : undefined; };
const OUT = arg("out") ?? join(ROOT, "dist"), PKG = join(OUT, "Chrysalis"), T = mkdtempSync(join(tmpdir(), "chrysalis-package-"));
const VERSION = arg("version") ?? ((await $`git -C ${ROOT} describe --tags --always`.nothrow().quiet().text()).trim() || "dev");
const PKGS = /^PKGS=\((.*)\)$/m.exec(readFileSync(join(ROOT, "chrysalis.sh"), "utf8"))![1].split(" ").map((p) => p.replace(/"/g, ""));
const say = (line: string) => console.log(`  ${line}`);

async function fetchTo(url: string, sha: string, to: string) {
  const got = new Uint8Array(await (await fetch(url)).arrayBuffer());
  if (new Bun.CryptoHasher("sha256").update(got).digest("hex") !== sha) throw new Error(`${basename(url)} does not match its checksum`);
  writeFileSync(to, got);
}

async function engines() {
  const from = arg("engines");
  if (from) return void cpSync(from, join(ROOT, "build"), { recursive: true });
  say("building the engines");
  if (WIN) await $`powershell -NoProfile -ExecutionPolicy Bypass -File ${join(ROOT, "windows", "build.ps1")} -Sdk ${arg("sdk") ?? process.env.HIP_PATH ?? ""}`.cwd(ROOT);
  else if (MAC) await $`make -C ${ROOT} -s engines`;
  else {
    rmSync(join(ROOT, "dist", "native"), { recursive: true, force: true });
    mkdirSync(join(ROOT, "dist", "native"), { recursive: true });
    await $`docker run --rm -e OWNER=${`${process.getuid!()}:${process.getgid!()}`} -v ${`${ROOT}:/src:ro`} -v ${`${join(ROOT, "dist", "native")}:/out`} quay.io/pypa/manylinux_2_28_x86_64 bash /src/chrysalis.sh native`;
  }
}

async function placeEngines() {
  const build = join(PKG, "build"), have = (f: string) => existsSync(join(ROOT, "build", f));
  mkdirSync(build, { recursive: true });
  if (WIN) {
    for (const f of readdirSync(join(ROOT, "build")).filter((f) => f.endsWith(".dll") || f === "ramdrive.exe" || f === "gpu.exe")) copyFileSync(join(ROOT, "build", f), join(build, f));
    if (have("hipblaslt")) cpSync(join(ROOT, "build", "hipblaslt"), join(build, "hipblaslt"), { recursive: true });
    await fetchTo("https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi", "073a70e00f77423e34bed98b86e600def93393ba5822204fac57a29324db9f7a", join(build, "winfsp.msi"));
  } else if (MAC) {
    for (const f of ["libchrysalis.dylib", "libchrysalis_fn.dylib", "chrysalis.metallib", "chrysalis_fn.metallib"]) copyFileSync(join(ROOT, "build", f), join(build, f));
  } else {
    const native = join(ROOT, "dist", "native"), R = join(ROOT, "data", "rocm", "lib");
    cpSync(join(native, "build"), build, { recursive: true });
    cpSync(join(native, "ggml"), join(PKG, "ggml"), { recursive: true });
    const libs = await $`ldd ${join(build, "libchrysalis.so")} ${join(build, "libchrysalis_fn.so")} ${join(build, "gpu")}`
      .env({ ...process.env, LD_LIBRARY_PATH: [R, join(R, "rocm_sysdeps", "lib"), join(native, "ggml", "avx2")].join(":") }).text();
    for (const lib of new Set([...libs.matchAll(/=> (\S+)/g)].map((m) => m[1]).filter((p) => p.startsWith(`${R}/`)))) {
      mkdirSync(dirname(join(PKG, "data", "rocm", "lib", lib.slice(R.length + 1))), { recursive: true });
      copyFileSync(lib, join(PKG, "data", "rocm", "lib", lib.slice(R.length + 1)));
    }
    cpSync(join(R, "hipblaslt", "library", "gfx1100"), join(PKG, "data", "rocm", "lib", "hipblaslt", "library", "gfx1100"), { recursive: true });
  }
  if (existsSync(join(ROOT, "data", "cache", "hipblaslt.txt")) && !MAC) {
    mkdirSync(join(PKG, "data", "cache"), { recursive: true });
    copyFileSync(join(ROOT, "data", "cache", "hipblaslt.txt"), join(PKG, "data", "cache", "hipblaslt.txt"));
  }
}

function placeCode() {
  cpSync(join(ROOT, "chrysalis"), join(PKG, "chrysalis"), { recursive: true, filter: (p) => basename(p) !== "__pycache__" });
  copyFileSync(join(ROOT, "LICENSE"), join(PKG, "LICENSE"));
  writeFileSync(join(PKG, "VERSION"), `${VERSION}\n`);
  const models = join(PKG, "data", "models");
  for (const f of ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "qwen3.8-27b/config.json", "qwen3.8-27b/generation_config.json",
    "qwen3.8-27b/preprocessor_config.json", "qwen3.8-flash-next/preprocessor_config.json"]) {
    mkdirSync(dirname(join(models, f)), { recursive: true });
    copyFileSync(join(ROOT, "data", "models", f), join(models, f));
  }
}

async function placePython() {
  say("putting Python and its packages in");
  const uv = WIN ? ["x86_64-pc-windows-msvc.zip", "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a"]
    : MAC ? ["aarch64-apple-darwin.tar.gz", "743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73"]
    : ["x86_64-unknown-linux-gnu.tar.gz", "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f"];
  const archive = join(T, `uv-${uv[0]}`);
  await fetchTo(`https://github.com/astral-sh/uv/releases/download/0.13.5/uv-${uv[0]}`, uv[1], archive);
  mkdirSync(join(T, "uv"));
  if (WIN) await $`tar -xf ${archive} -C ${join(T, "uv")}`;
  else await $`tar -xzf ${archive} -C ${join(T, "uv")} --strip-components=1`;
  const bin = join(T, "uv", WIN ? "uv.exe" : "uv");
  await $`${bin} python install -q --no-bin 3.12 --install-dir ${join(T, "python")}`;
  const venv = join(PKG, "data", "venv");
  renameSync(join(T, "python", readdirSync(join(T, "python")).find((d) => d.startsWith("cpython-3.12."))!), venv);
  const py = WIN ? join(venv, "python.exe") : join(venv, "bin", "python3");
  await $`${bin} pip install -q --python ${py} --break-system-packages ${PKGS}`;
  if (!WIN && !existsSync(join(venv, "bin", "python"))) symlinkSync("python3", join(venv, "bin", "python"));
  const lib = WIN ? join(venv, "Lib") : join(venv, "lib", "python3.12");
  for (const d of [...(WIN ? ["tcl", "include", "libs", "Scripts"].map((d) => join(venv, d)) : [join(venv, "include")]),
    ...["idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip", join("site-packages", "pip"), ...readdirSync(join(lib, WIN ? "" : "lib-dynload")).filter((f) => f.startsWith("_tkinter")).map((f) => join(WIN ? "" : "lib-dynload", f))].map((d) => join(lib, d)),
    ...(WIN ? [] : readdirSync(join(venv, "lib")).filter((d) => /^(tcl|tk|itcl|thread)\d/.test(d)).map((d) => join(venv, "lib", d)))]) rmSync(d, { recursive: true, force: true });
}

async function placeWeb() {
  say("bundling the web chat");
  const web = join(PKG, "web");
  mkdirSync(web, { recursive: true });
  for (const f of readdirSync(join(ROOT, "web")))
    if (statSync(join(ROOT, "web", f)).isFile() && /\.(ts|tsx|html|css|js|svg|json|lock|md)$/.test(f) && !/\.test\.ts$|\.check\.ts$|^fake-|^bench-|-harness\.ts$|^tests\.ts$/.test(f))
      copyFileSync(join(ROOT, "web", f), join(web, f));
  await $`${process.execPath} install --frozen-lockfile`.cwd(join(ROOT, "web")).quiet();
  await $`${process.execPath} build server.ts cli.ts --target=bun --production --external react-devtools-core --outdir ${web}`.cwd(join(ROOT, "web")).quiet();
  for (const p of ["mermaid/dist/mermaid.esm.min.mjs", "mermaid/dist/chunks/mermaid.esm.min", "aframe/dist/aframe-v1.8.0.min.js", "three/package.json", "three/build", "three/examples/jsm"])
    cpSync(join(ROOT, "web", "node_modules", p), join(web, "node_modules", p), { recursive: true, filter: (f) => !f.endsWith(".map") });
  mkdirSync(join(web, "node_modules", "react-devtools-core"), { recursive: true });
  writeFileSync(join(web, "node_modules", "react-devtools-core", "package.json"), '{ "name": "react-devtools-core", "type": "module", "main": "index.js" }\n');
  writeFileSync(join(web, "node_modules", "react-devtools-core", "index.js"), "export default { initialize() {}, connectToDevTools() {} };\n");
}

const CRC = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc = (b: Uint8Array) => { let c = ~0; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return ~c >>> 0; };
function png(size: number): Buffer {
  const svg = readFileSync(join(ROOT, "web", "logo.svg"), "utf8"), box = /viewBox="([^"]+)"/.exec(svg)![1].split(/\s+/).map(Number);
  const fill = /fill="#([0-9a-f]{6})"/i.exec(svg)![1], [r, g, b] = [0, 2, 4].map((i) => parseInt(fill.slice(i, i + 2), 16));
  const shapes = /\sd="([^"]+)"/.exec(svg)![1].split("M").filter(Boolean).map((p) => (p.match(/-?[\d.]+/g) ?? []).map(Number));
  const S = 4, k = (size * S * 880) / 1024 / Math.max(box[2], box[3]), ox = (size * S - k * box[2]) / 2 - k * box[0], oy = (size * S - k * box[3]) / 2 - k * box[1];
  const cover = new Float32Array(size * size);
  for (let y = 0; y < size * S; y++) {
    const at = y + 0.5, cuts: number[] = [];
    for (const s of shapes) for (let i = 0; i < s.length; i += 2) {
      const j = (i + 2) % s.length, y0 = s[i + 1] * k + oy, y1 = s[j + 1] * k + oy;
      if ((y0 <= at) !== (y1 <= at)) cuts.push(s[i] * k + ox + ((at - y0) / (y1 - y0)) * (s[j] - s[i]) * k);
    }
    cuts.sort((p, q) => p - q);
    for (let i = 0; i + 1 < cuts.length; i += 2) for (let x = Math.max(0, Math.ceil(cuts[i] - 0.5)); x < Math.min(size * S, cuts[i + 1] - 0.5); x++) cover[Math.floor(y / S) * size + Math.floor(x / S)] += 1 / (S * S);
  }
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) raw.set([r, g, b, Math.round(Math.min(1, cover[y * size + x]) * 255)], y * (size * 4 + 1) + 1 + x * 4);
  const chunk = (type: string, data: Buffer) => { const head = Buffer.alloc(8), tail = Buffer.alloc(4); head.writeUInt32BE(data.length); head.write(type, 4); tail.writeUInt32BE(crc(Buffer.concat([head.subarray(4), data]))); return Buffer.concat([head, data, tail]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(size); ihdr.writeUInt32BE(size, 4); ihdr.set([8, 6, 0, 0, 0], 8);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}
const ICO = [16, 20, 24, 32, 40, 48, 64, 256];
function ico(): { file: Buffer; group: Buffer } {
  const pngs = ICO.map(png), head = Buffer.alloc(6), group = Buffer.alloc(6 + 14 * ICO.length), entries = Buffer.alloc(16 * ICO.length);
  head.writeUInt16LE(1, 2); head.writeUInt16LE(ICO.length, 4); head.copy(group);
  let at = 6 + 16 * ICO.length;
  ICO.forEach((s, i) => {
    for (const [buf, o] of [[entries, 16 * i], [group, 6 + 14 * i]] as const) { buf[o] = s % 256; buf[o + 1] = s % 256; buf.writeUInt16LE(1, o + 4); buf.writeUInt16LE(32, o + 6); buf.writeUInt32LE(pngs[i].length, o + 8); }
    entries.writeUInt32LE(at, 16 * i + 12); group.writeUInt16LE(i + 1, 6 + 14 * i + 12); at += pngs[i].length;
  });
  return { file: Buffer.concat([head, entries, ...pngs]), group };
}
function icns(): Buffer {
  const parts = ([["ic07", 128], ["ic08", 256], ["ic09", 512], ["ic10", 1024]] as const).map(([type, size]) => { const p = png(size), h = Buffer.alloc(8); h.write(type); h.writeUInt32BE(8 + p.length, 4); return Buffer.concat([h, p]); });
  const head = Buffer.alloc(8); head.write("icns"); head.writeUInt32BE(8 + parts.reduce((n, p) => n + p.length, 0), 4);
  return Buffer.concat([head, ...parts]);
}

async function launcher(): Promise<string> {
  say("compiling the launcher");
  const exe = WIN ? join(PKG, "Chrysalis.exe") : MAC ? join(T, "Chrysalis") : join(PKG, "Chrysalis"), extra: string[] = [];
  if (WIN) {
    writeFileSync(join(T, "chrysalis.ico"), ico().file);
    extra.push(`--windows-icon=${join(T, "chrysalis.ico")}`, "--windows-title=Chrysalis", "--windows-description=Chrysalis", "--windows-publisher=Chrysalis Inference");
  }
  if (MAC) {
    const bun = join(T, "bun"), b = Buffer.from(readFileSync(process.execPath));
    for (let i = 0, at = 32; i < b.readUInt32LE(16); i++, at += b.readUInt32LE(at + 4)) if (b.readUInt32LE(at) === 0x1b) crypto.getRandomValues(b.subarray(at + 8, at + 24));
    writeFileSync(bun, b, { mode: 0o755 });
    extra.push(`--compile-executable-path=${bun}`);
  }
  await $`${process.execPath} build --compile ${extra} ${join(ROOT, "web", "launch.ts")} --outfile ${exe}`.quiet();
  if (WIN) {
    const { dlopen, FFIType, ptr } = await import("bun:ffi");
    const k = dlopen("kernel32.dll", { BeginUpdateResourceW: { args: [FFIType.ptr, FFIType.bool], returns: FFIType.ptr }, UpdateResourceW: { args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.u16, FFIType.ptr, FFIType.u32], returns: FFIType.bool }, EndUpdateResourceW: { args: [FFIType.ptr, FFIType.bool], returns: FFIType.bool } });
    const wide = (s: string) => Buffer.from(`${s}\0`, "utf16le"), group = ico().group, update = k.symbols.BeginUpdateResourceW(ptr(wide(exe)), false);
    if (!update || !k.symbols.UpdateResourceW(update, 14 as never, ptr(wide("IDI_MYICON")), 1033, ptr(group), group.length) || !k.symbols.EndUpdateResourceW(update, false)) throw new Error("Chrysalis.exe's icon was not set");
  }
  return exe;
}

async function wrapWindows() {
  let iscc = join(process.env.LOCALAPPDATA ?? "", "Programs", "Inno Setup 6", "ISCC.exe");
  if (!existsSync(iscc)) {
    await fetchTo("https://github.com/jrsoftware/issrc/releases/download/is-6_7_4/innosetup-6.7.4.exe", "a549929319f9120716677c685535ca4671e4b937b5fdc87e2585f32d8eb30e8c", join(T, "innosetup.exe"));
    await $`${join(T, "innosetup.exe")} /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CURRENTUSER`;
  }
  await $`${iscc} /Q ${`/DRoot=${PKG}`} ${`/DVersion=${VERSION}`} ${`/DIcon=${join(T, "chrysalis.ico")}`} ${`/O${OUT}`} ${join(ROOT, "windows", "setup.iss")}`;
  return join(OUT, "Chrysalis-Setup.exe");
}

async function wrapMac(exe: string) {
  const app = join(OUT, "Chrysalis.app"), contents = join(app, "Contents"), resources = join(contents, "Resources"), n = VERSION.replace(/^v/, "").split("-")[0];
  rmSync(app, { recursive: true, force: true });
  mkdirSync(join(contents, "MacOS"), { recursive: true });
  renameSync(exe, join(contents, "MacOS", "Chrysalis"));
  renameSync(PKG, resources);
  writeFileSync(join(resources, "chrysalis.icns"), icns());
  writeFileSync(join(contents, "Info.plist"), ['<?xml version="1.0" encoding="UTF-8"?>', '<plist version="1.0"><dict>',
    "<key>CFBundleIdentifier</key><string>io.github.michaeldinelle.chrysalis</string>", "<key>CFBundleName</key><string>Chrysalis</string>",
    "<key>CFBundleExecutable</key><string>Chrysalis</string>", "<key>CFBundlePackageType</key><string>APPL</string>",
    `<key>CFBundleShortVersionString</key><string>${n}</string>`, `<key>CFBundleVersion</key><string>${n}</string>`, "<key>LSMinimumSystemVersion</key><string>13.0</string>",
    "<key>CFBundleIconFile</key><string>chrysalis</string>", "<key>LSBackgroundOnly</key><true/>",
    "<key>NSLocalNetworkUsageDescription</key><string>Chrysalis finds the computer on your network that runs it, and answers from its GPU.</string>", "</dict></plist>", ""].join("\n"));
  const rights = join(T, "rights.plist");
  writeFileSync(rights, ['<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>', ...["allow-jit", "allow-unsigned-executable-memory", "disable-executable-page-protection", "allow-dyld-environment-variables", "disable-library-validation"]
    .map((r) => `<key>com.apple.security.cs.${r}</key><true/>`), "</dict></plist>"].join(""));
  const id = /\s([0-9A-F]{40}) "(Developer ID Application|Apple Development): /.exec(await $`security find-identity -v -p codesigning`.nothrow().text())?.[1] ?? "-";
  const sign = (path: string) => $`codesign --force --options runtime --timestamp=none --entitlements ${rights} -s ${id} ${path}`.quiet();
  const machO = (path: string) => {
    const head = Buffer.alloc(4), fd = openSync(path, "r");
    try { return readSync(fd, head, 0, 4, 0) === 4 && [0xcffaedfe, 0xcafebabe, 0xfeedfacf].includes(head.readUInt32BE(0)); } finally { closeSync(fd); }
  };
  const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((e) => e.isSymbolicLink() ? [] : e.isDirectory() ? walk(join(dir, e.name)) : machO(join(dir, e.name)) ? [join(dir, e.name)] : []);
  say(`signing ${id === "-" ? "for this Mac alone (no signing certificate in the keychain)" : "with this Mac's certificate"}`);
  for (const f of walk(resources)) await sign(f);
  await sign(join(contents, "MacOS", "Chrysalis"));
  await sign(app);
  await $`codesign -v --deep --strict ${app}`;
  const team = /^TeamIdentifier=(\S+)$/m.exec(await $`codesign -dv ${app} 2>&1`.nothrow().text())?.[1];
  if (team && team !== "not") await notarize(app, team, n);
  else say("no Apple developer team in the keychain: the app is signed for this Mac alone, and others open it only after Open Anyway");
  const stage = join(T, "dmg");
  mkdirSync(stage);
  cpSync(app, join(stage, "Chrysalis.app"), { recursive: true, verbatimSymlinks: true });
  symlinkSync("/Applications", join(stage, "Applications"));
  const dmg = join(OUT, "Chrysalis.dmg");
  await $`hdiutil create -quiet -volname Chrysalis -srcfolder ${stage} -format ULMO -ov ${dmg}`;
  return dmg;
}

async function notarize(app: string, team: string, n: string) {
  const plist = (body: string) => `<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>${body}</dict></plist>`;
  const archive = join(T, "Chrysalis.xcarchive");
  mkdirSync(join(archive, "Products", "Applications"), { recursive: true });
  await $`ditto ${app} ${join(archive, "Products", "Applications", "Chrysalis.app")}`;
  writeFileSync(join(archive, "Info.plist"), plist(`<key>ApplicationProperties</key><dict><key>ApplicationPath</key><string>Applications/Chrysalis.app</string>
<key>Architectures</key><array><string>arm64</string></array><key>CFBundleIdentifier</key><string>io.github.michaeldinelle.chrysalis</string>
<key>CFBundleShortVersionString</key><string>${n}</string><key>CFBundleVersion</key><string>${n}</string><key>SigningIdentity</key><string>Apple Development</string>
<key>Team</key><string>${team}</string></dict><key>ArchiveVersion</key><integer>2</integer><key>CreationDate</key><date>${new Date().toISOString().slice(0, 19)}Z</date>
<key>Name</key><string>Chrysalis</string><key>SchemeName</key><string>Chrysalis</string>`));
  const options = (to: string) => {
    const file = join(T, `${to}.plist`);
    writeFileSync(file, plist(`<key>method</key><string>developer-id</string><key>signingStyle</key><string>automatic</string><key>teamID</key><string>${team}</string><key>destination</key><string>${to}</string>`));
    return file;
  };
  const xcode = async (args: string[], log: string) => {
    const r = await $`xcodebuild ${args} -allowProvisioningUpdates`.nothrow().quiet();
    writeFileSync(join(T, log), r.stdout.toString() + r.stderr.toString());
    return r.exitCode === 0;
  };
  say(`signing with Developer ID for team ${team}`);
  if (!(await xcode(["-exportArchive", "-archivePath", archive, "-exportOptionsPlist", options("export"), "-exportPath", join(T, "signed")], "signed.log")))
    throw new Error(`Developer ID signing failed: ${readFileSync(join(T, "signed.log"), "utf8").split("\n").filter((l) => /error/i.test(l)).slice(0, 3).join(" ")}`);
  say("sending it to Apple to notarize");
  if (!(await xcode(["-exportArchive", "-archivePath", archive, "-exportOptionsPlist", options("upload"), "-exportPath", join(T, "upload")], "upload.log")))
    throw new Error(`Apple did not take it for notarizing: ${readFileSync(join(T, "upload.log"), "utf8").split("\n").filter((l) => /error/i.test(l)).slice(0, 3).join(" ")}`);
  const done = join(T, "notarized", "Chrysalis.app");
  for (;;) {
    const ok = await xcode(["-exportNotarizedApp", "-archivePath", archive, "-exportPath", join(T, "notarized")], "notarized.log");
    if (ok && existsSync(done)) break;
    const log = readFileSync(join(T, "notarized.log"), "utf8");
    if (!/is processing|in progress/i.test(log)) throw new Error(`Apple did not notarize it: ${log.split("\n").filter((l) => /error/i.test(l)).slice(0, 3).join(" ")}`);
    await Bun.sleep(20_000);
  }
  if ((await $`xcrun stapler validate -q ${done}`.nothrow().quiet()).exitCode) await $`xcrun stapler staple -q ${done}`;
  rmSync(app, { recursive: true, force: true });
  await $`ditto ${done} ${app}`;
  await $`spctl --assess --type execute ${app}`;
  say("notarized by Apple, its ticket stapled to the app");
}

async function wrapLinux(exe: string) {
  const tar = join(OUT, "Chrysalis-linux-x86_64.tar.gz");
  chmodSync(exe, 0o755);
  await $`tar -C ${OUT} -czf ${tar} Chrysalis`;
  return tar;
}

rmSync(PKG, { recursive: true, force: true });
mkdirSync(PKG, { recursive: true });
try {
  await engines();
  await placeEngines();
  placeCode();
  await placePython();
  await placeWeb();
  const exe = await launcher();
  const runs = async (path: string) => { if ((await $`${path} help`.nothrow().quiet()).exitCode) throw new Error("the launcher does not run"); };
  if (!MAC) await runs(exe);
  const made = WIN ? await wrapWindows() : MAC ? await wrapMac(exe) : await wrapLinux(exe);
  if (MAC) await runs(join(OUT, "Chrysalis.app", "Contents", "MacOS", "Chrysalis"));
  say(`${made}: ${(statSync(made).size / 1e6).toFixed(0)} MB, ${VERSION}`);
} finally {
  rmSync(T, { recursive: true, force: true });
}
