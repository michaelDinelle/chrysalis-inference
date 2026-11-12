import logo from "./logo.svg";

const SEA = " ·~≈+*#";
const TUBE = "·:+*≈#%@";
const ROW = 32;
const CELL_W = 6, CELL_H = 11, FONT_PX = 10;
const VIEW_W = 660, VIEW_H = 473, HALF_STROKE = 11.26;
const MIN_HALF_STROKE = 9;
const strokes = (w: number) => { const half = HALF_STROKE * w / VIEW_W; return { half: Math.max(half, MIN_HALF_STROKE), bold: Math.max(0, MIN_HALF_STROKE - half) }; };
const PATTERNS = [1, 0, 2, 3, 4];
const WAKE = 9, MAX_RIPPLES = 16, WAVE_LIFE = 5, WAKE_LIFE = 2;
const HOVER = 0.033;
const SINK = 0.34, RISE = 0.26;

const VERT = `#version 300 es
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const CELLS = `#version 300 es
precision highp float;
uniform vec2 uGrid, uCell;
uniform vec4 uView;
uniform vec4 uBox;
uniform vec2 uLane;
uniform float uFlyOn;
uniform float uTime, uMotion;
uniform int uWaves;
uniform vec4 uWave[${2 * MAX_RIPPLES}];
uniform vec3 uFly;
uniform vec2 uCenter;
uniform sampler2D uSdf;
uniform vec2 uSdfSize;
uniform vec3 uPointer;
out vec4 o;

ivec4 permute(ivec4 x) { return ((x * 34 + 1) * x) % 289; }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
  ivec3 n = ivec3(mod(i, 289.0)), n1 = ivec3(i1), n2 = ivec3(i2);
  ivec4 p = permute(permute(permute(n.z + ivec4(0, n1.z, n2.z, 1)) + n.y + ivec4(0, n1.y, n2.y, 1)) + n.x + ivec4(0, n1.x, n2.x, 1));
  ivec4 j = p % 49, xi = 4 * (j / 7) - 13, yi = 4 * (j % 7) - 13;
  vec4 x = vec4(xi) / 14.0;
  vec4 y = vec4(yi) / 14.0;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -vec4(greaterThanEqual(abs(xi) + abs(yi), ivec4(14)));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
float fbm(vec3 p) { return 0.6 * snoise(p) + 0.28 * snoise(p * 2.03 + 17.0) + 0.12 * snoise(p * 4.1 + 41.0); }

float sea(vec2 p, float t) {
  vec2 s = p * 0.0038;
  vec2 q = vec2(fbm(vec3(s, t * 0.04)), fbm(vec3(s + vec2(5.2, 1.3), t * 0.04)));
  float n = fbm(vec3(s * 1.6 + 1.8 * q, t * 0.06));
  float ribbon = 1.0 - abs(n);
  ribbon *= ribbon;
  ribbon *= ribbon * ribbon;
  float swell = 0.5 + 0.5 * sin(p.x * 0.011 + p.y * 0.019 - t * 0.5 + 2.4 * q.x);
  return clamp(0.06 + 0.7 * ribbon + 0.16 * swell * ribbon + 0.08 * swell, 0.0, 1.0);
}

float dot2(vec2 v) { return dot(v, v); }
float sdHeart(vec2 p) {
  p.x = abs(p.x);
  if (p.y + p.x > 1.0) return sqrt(dot2(p - vec2(0.25, 0.75))) - sqrt(2.0) / 4.0;
  return sqrt(min(dot2(p - vec2(0.0, 1.0)), dot2(p - 0.5 * max(p.x + p.y, 0.0)))) * sign(p.x - p.y);
}
float sdStar5(vec2 p, float r, float rf) {
  const vec2 k1 = vec2(0.809016994375, -0.587785252292);
  const vec2 k2 = vec2(-k1.x, k1.y);
  p.x = abs(p.x);
  p -= 2.0 * max(dot(k1, p), 0.0) * k1;
  p -= 2.0 * max(dot(k2, p), 0.0) * k2;
  p.x = abs(p.x);
  p.y -= r;
  vec2 ba = rf * vec2(-k1.y, k1.x) - vec2(0.0, 1.0);
  float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
  return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

float flyAt(vec2 p, vec2 c, float w) {
  float h = w * ${VIEW_H / VIEW_W};
  vec2 uv = (p - c) / vec2(w, h) + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1e4;
  return texture(uSdf, uv).r * (w / uSdfSize.x);
}

float shapeAt(float kind, vec2 d, float r) {
  if (kind < 0.5 || kind > 8.5) return length(d) - r;
  if (kind < 1.5) return sdHeart(vec2(d.x, -d.y) / r * 0.8 + vec2(0.0, 0.55)) * r / 0.8;
  if (kind < 2.5) return flyAt(d, vec2(0.0), r * 2.2);
  if (kind < 3.5) return sdStar5(d * vec2(1.0, -1.0) / r, 1.0, 0.45) * r;
  float a = atan(d.y, d.x);
  return length(d) - r * (0.78 + 0.22 * cos(6.0 * a));
}

float waves(vec2 p) {
  float sum = 0.0;
  for (int i = 0; i < uWaves; i++) {
    vec4 a = uWave[2 * i];
    vec2 b = uWave[2 * i + 1].xy;
    float x = shapeAt(a.z, p - a.xy, b.x) / b.y;
    sum += a.w * cos(6.2832 * x) * exp(-0.9 * (x + 0.6) * (x + 0.6));
  }
  return sum;
}
float grain(vec2 p, vec2 cell) {
  return 0.999 * (0.65 * (0.5 + 0.5 * snoise(vec3(p * 0.018, 5.0))) + 0.35 * fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453));
}

float boxAt(vec2 p) {
  vec2 q = abs(p - 0.5 * (uBox.xy + uBox.zw)) - 0.5 * (uBox.zw - uBox.xy) + 20.0;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 20.0;
}

void main() {
  vec2 cell = vec2(gl_FragCoord.x - 0.5, uGrid.y - 0.5 - gl_FragCoord.y);
  vec2 p = (cell + 0.5) * uCell;
  float t = uTime * uMotion;
  float w = 0.0;
  vec2 slope = vec2(0.0);
  if (uWaves > 0) {
    w = waves(p);
    slope = vec2(waves(p + vec2(3.0, 0.0)) - w, waves(p + vec2(0.0, 3.0)) - w) / 3.0;
  }
  float breathe = 1.0 + 0.018 * sin(t * 0.9) * uMotion;
  float sunk = smoothstep(0.0, 1.0, 1.0 - uFlyOn);
  vec2 drift = sunk > 0.0 ? sunk * 6.0 * vec2(snoise(vec3(p * 0.011, t * 0.12)), snoise(vec3(p * 0.011 + 31.0, t * 0.12))) : vec2(0.0);
  vec2 q = uCenter + (p - uCenter) / vec2(breathe, 1.0) - slope * 30.0 + drift;
  float d = flyAt(q, uCenter, uFly.x) - uFly.z;
  float edge = smoothstep(0.0, 28.0, min(min(p.x - uView.x, uView.z - p.x), min(p.y - uView.y, uView.w - p.y)));
  float hover = uPointer.z * exp(-dot2(p - uPointer.xy) / 3600.0);
  float lift = max(w, 0.0);
  if (d < 0.5 && uFlyOn > 0.0 && (uFlyOn >= 1.0 || grain(p, cell) < uFlyOn)) {
    float depth = clamp(-d / uFly.y, 0.0, 1.0);
    float prof = sqrt(max(0.0, 1.0 - (1.0 - depth) * (1.0 - depth)));
    float g = ${ROW}.0 + floor(clamp(1.0 + prof * ${TUBE.length - 2}.0 + 0.5 + lift * 0.5, 0.0, ${TUBE.length - 1}.0));
    o = vec4(g / 255.0, (0.58 + 0.42 * prof) * (0.4 + 0.6 * uFlyOn), 1.0, clamp(max(0.0, prof - 0.55) * 1.4 * uFlyOn + lift * 0.3, 0.0, 1.0));
    return;
  }
  float s = sea(p - slope * 90.0, t);
  float ghost = 1.0 - smoothstep(-2.0, 6.0, d);
  float gd = max(d, 0.0) / 30.0;
  float glow = exp(-gd * gd) * uFlyOn;
  float calm = mix(1.0, mix(0.3, 1.0, smoothstep(3.0, 40.0, d)), uFlyOn);
  float quiet = mix(0.3, 1.0, smoothstep(0.0, 40.0, boxAt(p)));
  quiet *= mix(mix(0.3, 0.65, smoothstep(0.0, 72.0, max(uLane.x - p.x, p.x - uLane.y))), 1.0, uFlyOn);
  float v = clamp(s * calm + ghost * 0.3 + lift * 0.55 + min(w, 0.0) * 0.15 + hover * 0.3, 0.0, 1.0);
  float g = floor(clamp(v * 3.9, 0.0, ${SEA.length - 1}.0));
  float bright = ((0.05 + 0.46 * v * v + 0.3 * glow + 0.3 * lift + 0.3 * hover) * quiet + 0.2 * ghost * sqrt(quiet)) * edge;
  o = vec4(g / 255.0, clamp(bright, 0.0, 1.0), clamp(glow * 0.85 + ghost * 0.7 + lift * 0.6 + hover * 0.5, 0.0, 1.0), 0.0);
}`;

const GLYPHS = `#version 300 es
precision highp float;
uniform sampler2D uCells, uAtlas;
uniform vec2 uCellPx;
uniform float uRows, uHeight;
uniform vec3 uSea, uPink, uHi;
out vec4 o;
void main() {
  vec2 px = vec2(gl_FragCoord.x, uHeight - gl_FragCoord.y);
  ivec2 cell = ivec2(px / uCellPx);
  vec4 c = texelFetch(uCells, ivec2(cell.x, int(uRows) - 1 - cell.y), 0);
  int g = int(c.r * 255.0 + 0.5);
  if (g == 0) discard;
  ivec2 local = ivec2(px) - cell * ivec2(uCellPx);
  float a = texelFetch(uAtlas, ivec2((g % ${ROW}) * int(uCellPx.x) + local.x, (g / ${ROW}) * int(uCellPx.y) + local.y), 0).a * c.g;
  vec3 col = mix(mix(uSea, uPink, c.b), uHi, c.a);
  o = vec4(col * a, a);
}`;

type Sdf = { data: Float32Array; w: number; h: number };

function edt1(f: Float64Array, n: number, v: Int32Array, z: Float64Array, d: Float64Array) {
  let k = 0;
  v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) { k--; s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
}
function edt(mask: Uint8Array, w: number, h: number, want: number): Float64Array {
  const out = new Float64Array(w * h), n = Math.max(w, h);
  const f = new Float64Array(n), d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
  for (let i = 0; i < w * h; i++) out[i] = mask[i] === want ? 0 : 1e20;
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = out[y * w + x];
    edt1(f, h, v, z, d);
    for (let y = 0; y < h; y++) out[y * w + x] = d[y];
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = out[y * w + x];
    edt1(f, w, v, z, d);
    for (let x = 0; x < w; x++) out[y * w + x] = d[x];
  }
  return out;
}

async function drawingSdf(width: number): Promise<Sdf> {
  const img = new Image();
  img.src = logo;
  await img.decode();
  const w = width, h = Math.round(width * VIEW_H / VIEW_W);
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data, mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = px[i * 4 + 3] > 127 ? 1 : 0;
  const outside = edt(mask, w, h, 1), inside = edt(mask, w, h, 0), data = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) data[i] = mask[i] ? -(Math.sqrt(inside[i]) - 0.5) : Math.sqrt(outside[i]) - 0.5;
  return { data, w, h };
}

const fract = (x: number) => x - Math.floor(x);
const clamp = (x: number, a: number, b: number) => Math.min(Math.max(x, a), b);
const smoothstep = (a: number, b: number, x: number) => { const k = clamp((x - a) / (b - a), 0, 1); return k * k * (3 - 2 * k); };
const mod289 = (x: number) => x - Math.floor(x / 289) * 289;
const permute = (x: number) => ((x * 34 + 1) * x) % 289;
function corner(p: number, x: number, y: number, z: number): number {
  let m = 0.6 - (x * x + y * y + z * z);
  if (m <= 0) return 0;
  const j = p % 49, xi = 4 * Math.floor(j / 7) - 13, yi = 4 * (j % 7) - 13;
  let gx = xi / 14, gy = yi / 14;
  const h = 1 - Math.abs(gx) - Math.abs(gy);
  if (Math.abs(xi) + Math.abs(yi) >= 14) { gx -= Math.floor(gx) * 2 + 1; gy -= Math.floor(gy) * 2 + 1; }
  const r = gx * gx + gy * gy + h * h, norm = 1.79284291400159 - 0.85373472095314 * r;
  m *= m;
  return m * m * norm * (gx * x + gy * y + h * z);
}
function snoise(vx: number, vy: number, vz: number): number {
  const s = (vx + vy + vz) / 3;
  let ix = Math.floor(vx + s), iy = Math.floor(vy + s), iz = Math.floor(vz + s);
  const u = (ix + iy + iz) / 6, x0 = vx - ix + u, y0 = vy - iy + u, z0 = vz - iz + u;
  const gx = x0 >= y0 ? 1 : 0, gy = y0 >= z0 ? 1 : 0, gz = z0 >= x0 ? 1 : 0;
  const i1x = Math.min(gx, 1 - gz), i1y = Math.min(gy, 1 - gx), i1z = Math.min(gz, 1 - gy);
  const i2x = Math.max(gx, 1 - gz), i2y = Math.max(gy, 1 - gx), i2z = Math.max(gz, 1 - gy);
  ix = mod289(ix); iy = mod289(iy); iz = mod289(iz);
  const c = 1 / 6;
  return 42 * (corner(permute(permute(permute(iz) + iy) + ix), x0, y0, z0)
    + corner(permute(permute(permute(iz + i1z) + iy + i1y) + ix + i1x), x0 - i1x + c, y0 - i1y + c, z0 - i1z + c)
    + corner(permute(permute(permute(iz + i2z) + iy + i2y) + ix + i2x), x0 - i2x + 2 * c, y0 - i2y + 2 * c, z0 - i2z + 2 * c)
    + corner(permute(permute(permute(iz + 1) + iy + 1) + ix + 1), x0 - 0.5, y0 - 0.5, z0 - 0.5));
}
const fbm = (x: number, y: number, z: number) =>
  0.6 * snoise(x, y, z) + 0.28 * snoise(x * 2.03 + 17, y * 2.03 + 17, z * 2.03 + 17) + 0.12 * snoise(x * 4.1 + 41, y * 4.1 + 41, z * 4.1 + 41);
function sea(px: number, py: number, t: number): number {
  const sx = px * 0.0038, sy = py * 0.0038;
  const qx = fbm(sx, sy, t * 0.04), qy = fbm(sx + 5.2, sy + 1.3, t * 0.04);
  const n = fbm(sx * 1.6 + 1.8 * qx, sy * 1.6 + 1.8 * qy, t * 0.06);
  let ribbon = 1 - Math.abs(n);
  ribbon *= ribbon;
  ribbon *= ribbon * ribbon;
  const swell = 0.5 + 0.5 * Math.sin(px * 0.011 + py * 0.019 - t * 0.5 + 2.4 * qx);
  return clamp(0.06 + 0.7 * ribbon + 0.16 * swell * ribbon + 0.08 * swell, 0, 1);
}
function sdHeart(x: number, y: number): number {
  x = Math.abs(x);
  if (y + x > 1) return Math.hypot(x - 0.25, y - 0.75) - Math.SQRT2 / 4;
  const k = 0.5 * Math.max(x + y, 0);
  return Math.sqrt(Math.min(x * x + (y - 1) * (y - 1), (x - k) * (x - k) + (y - k) * (y - k))) * Math.sign(x - y);
}
function sdStar5(x: number, y: number, r: number, rf: number): number {
  const k1x = 0.809016994375, k1y = -0.587785252292, k2x = -k1x, k2y = k1y;
  x = Math.abs(x);
  let d = 2 * Math.max(k1x * x + k1y * y, 0);
  x -= d * k1x; y -= d * k1y;
  d = 2 * Math.max(k2x * x + k2y * y, 0);
  x -= d * k2x; y -= d * k2y;
  x = Math.abs(x);
  y -= r;
  const bax = rf * -k1y, bay = rf * k1x - 1, h = clamp((x * bax + y * bay) / (bax * bax + bay * bay), 0, r);
  return Math.hypot(x - bax * h, y - bay * h) * Math.sign(y * bax - x * bay);
}
type Wave = { x: number; y: number; kind: number; amp: number; size: number; len: number };

class Cpu {
  cells = new Uint8ClampedArray(0);
  ink = new Uint8ClampedArray(0);
  inkW = 0;
  pixels: ImageData | null = null;
  constructor(readonly ctx: CanvasRenderingContext2D) {}
}

function compile(gl: WebGL2RenderingContext, frag: string): WebGLProgram {
  const prog = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, VERT], [gl.FRAGMENT_SHADER, frag]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`ocean shader: ${gl.getShaderInfoLog(s)}`);
    gl.attachShader(prog, s);
  }
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`ocean program: ${gl.getProgramInfoLog(prog)}`);
  return prog;
}

const rgb = (css: string) => {
  const [r = 136, g = 136, b = 136] = css.match(/[\d.]+/g)?.map(Number) ?? [];
  return [r / 255, g / 255, b / 255];
};

export class Ocean {
  canvas = document.createElement("canvas");
  private gl: WebGL2RenderingContext | null = null;
  private cpu: Cpu | null = null;
  private cost = 0;
  private cellsProg!: WebGLProgram;
  private glyphProg!: WebGLProgram;
  private u = new Map<string, WebGLUniformLocation | null>();
  private cellTex: WebGLTexture | null = null;
  private fbo: WebGLFramebuffer | null = null;
  private atlas: WebGLTexture | null = null;
  private sdfTex: WebGLTexture | null = null;
  private sdf: Sdf | null = null;
  private empty: HTMLElement | null = null;
  private shown = false;
  private fly = 0;
  private cols = 0;
  private rows = 0;
  private dpr = 0;
  private cellPx = [0, 0];
  private view = [0, 0, 0, 0];
  private seen = [0, 0, 0, 0];
  private box = [-1e4, -1e4, -1e4, -1e4];
  private lane = [0, 0];
  private center = [0, 0];
  private flyW = 0;
  private ripples = new Float32Array(MAX_RIPPLES * 4).fill(-1);
  private packed = new Float32Array(MAX_RIPPLES * 8);
  private pattern = 0;
  private t0 = performance.now() / 1000;
  private pointer = [0, 0, 0];
  private presence = 0;
  private moved = 0;
  private lastWake: [number, number] | null = null;
  private colors = { sea: [0, 0, 0], pink: [0, 0, 0], hi: [0, 0, 0] };
  private drawn = 0;
  private running = false;
  private still = matchMedia("(prefers-reduced-motion: reduce)");
  private resize = new ResizeObserver(() => this.layout());

  private boxAt: string;
  private hintAt: string;
  private roomAt: string;
  constructor(private layer: HTMLElement, { box = ".composer-box", hint = "#composer .hint", room = "" } = {}) {
    this.boxAt = box;
    this.hintAt = hint;
    this.roomAt = room;
    this.dress(this.canvas);
    this.still.addEventListener("change", () => this.wake());
    addEventListener("scroll", () => this.layout(), { passive: true });
  }

  private dress(c: HTMLCanvasElement) {
    c.className = "ocean";
    c.setAttribute("aria-hidden", "true");
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      this.splash(e.clientX, e.clientY, PATTERNS[this.pattern++ % PATTERNS.length]);
      this.lastWake = [e.clientX, e.clientY];
    });
    c.addEventListener("pointermove", (e) => {
      this.pointer[0] = e.clientX; this.pointer[1] = e.clientY; this.presence = 1; this.moved = this.now();
      if (e.buttons & 1 && this.lastWake && Math.hypot(e.clientX - this.lastWake[0], e.clientY - this.lastWake[1]) > 34) {
        this.splash(e.clientX, e.clientY, WAKE);
        this.lastWake = [e.clientX, e.clientY];
      }
      this.wake();
    });
    c.addEventListener("pointerleave", () => { this.presence = 0; this.lastWake = null; });
    c.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.gl = null; this.running = false; });
    c.addEventListener("webglcontextrestored", () => { if (this.init()) this.layout(); });
  }

  private init(): boolean {
    const gl = this.cpu ? null : this.canvas.getContext("webgl2", { premultipliedAlpha: true, antialias: false, alpha: true });
    try {
      if (gl) {
        this.cellsProg = compile(gl, CELLS);
        this.glyphProg = compile(gl, GLYPHS);
      }
    } catch (e) {
      console.error(e);
      const fresh = document.createElement("canvas");
      this.dress(fresh);
      this.canvas.replaceWith(fresh);
      this.canvas = fresh;
    }
    this.gl = gl && this.cellsProg ? gl : null;
    if (!this.gl) {
      const ctx = this.canvas.getContext("2d");
      if (!ctx) return false;
      this.cpu = new Cpu(ctx);
    }
    this.u.clear();
    this.cellTex = this.fbo = this.atlas = this.sdfTex = null;
    this.cols = this.rows = this.dpr = 0;
    if (this.sdf) this.uploadSdf();
    return true;
  }

  start(): boolean {
    if (!this.init()) return false;
    this.layer.append(this.canvas);
    this.resize.observe(this.layer);
    const box = this.boxAt ? document.querySelector(this.boxAt) : null;
    if (box) this.resize.observe(box);
    this.readColors();
    if (document.readyState !== "complete") addEventListener("load", () => { this.readColors(); this.wake(); }, { once: true });
    void drawingSdf(720).then((s) => { this.sdf = s; this.uploadSdf(); this.wake(); });
    return true;
  }

  place() { this.layout(); }

  butterfly(empty: HTMLElement) {
    this.empty = empty;
    this.wake();
  }

  private live() { return !!(this.gl || this.cpu) && this.shown; }
  private now() { return performance.now() / 1000 - this.t0; }

  private readColors() {
    const probe = document.body.appendChild(document.createElement("i"));
    const read = (v: string) => { probe.style.color = `var(${v})`; return rgb(getComputedStyle(probe).color); };
    this.colors = { sea: read("--muted"), pink: read("--accent"), hi: [1.0, 0.86, 0.92] };
    probe.remove();
  }

  private splash(x: number, y: number, kind: number) {
    if (this.still.matches) return;
    let slot = 0, end = Infinity;
    for (let i = 0; i < MAX_RIPPLES; i++) {
      const k = this.ripples[i * 4 + 3], e = k < 0 ? -Infinity : this.ripples[i * 4 + 2] + (k === WAKE ? WAKE_LIFE : WAVE_LIFE);
      if (e < end) { end = e; slot = i; }
    }
    this.ripples.set([x, y, this.now(), kind], slot * 4);
    this.wake();
  }

  private waves(t: number): Wave[] {
    const out: Wave[] = [];
    for (let i = 0; i < MAX_RIPPLES; i++) {
      const [x, y, at, kind] = this.ripples.subarray(i * 4, i * 4 + 4);
      if (kind < 0) continue;
      const age = t - at, wake = kind === WAKE, life = wake ? WAKE_LIFE : WAVE_LIFE;
      if (age < 0 || age > life) continue;
      const k = 1 - age / life, size = 8 + (wake ? 90 : 320) * (1 - Math.exp(-0.8 * age));
      out.push({ x, y, kind, amp: (wake ? 0.5 : 1) * k ** 1.6 * smoothstep(0, 0.3, age), size, len: wake ? 30 : 44 + 0.06 * size });
    }
    return out;
  }

  private uploadSdf() {
    const gl = this.gl, s = this.sdf;
    if (!gl || !s) return;
    this.sdfTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.sdfTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, s.w, s.h, 0, gl.RED, gl.FLOAT, s.data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private layout() {
    if (!this.gl && !this.cpu) return;
    const r = this.layer.getBoundingClientRect(), was = this.shown;
    this.shown = r.width >= 40 && r.height >= 40;
    if (!this.shown) return;
    const dpr = this.cpu ? Math.max(1, Math.min(devicePixelRatio || 1, Math.sqrt(2.5e6 / (innerWidth * innerHeight)))) : devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(CELL_W * dpr)), ch = Math.max(1, Math.round(CELL_H * dpr));
    const cols = Math.ceil(innerWidth * dpr / cw), rows = Math.ceil(innerHeight * dpr / ch);
    if (dpr !== this.dpr || cw !== this.cellPx[0] || ch !== this.cellPx[1]) this.buildAtlas(dpr, cw, ch);
    const resized = cols !== this.cols || rows !== this.rows || (this.gl && !this.cellTex);
    if (resized) {
      this.buildCells(cols, rows);
      this.canvas.width = cols * cw;
      this.canvas.height = rows * ch;
      Object.assign(this.canvas.style, { width: `${cols * cw / dpr}px`, height: `${rows * ch / dpr}px` });
    }
    Object.assign(this.canvas.style, { left: `${-r.left}px`, top: `${-r.top}px` });
    this.view = [r.left, r.top, r.right, r.bottom];
    const x0 = Math.max(0, Math.floor(r.left * dpr / cw)), y0 = Math.max(0, Math.floor(r.top * dpr / ch));
    this.seen = [x0, y0, Math.min(cols, Math.ceil(r.right * dpr / cw)) - x0, Math.min(rows, Math.ceil(r.bottom * dpr / ch)) - y0];
    const box = this.boxAt ? document.querySelector(this.boxAt)?.getBoundingClientRect() : undefined;
    const hint = this.hintAt ? document.querySelector(this.hintAt)?.getBoundingClientRect() : undefined;
    const column = box && box.width > 0 ? box : r, top = box && box.width > 0 ? box.top : r.bottom;
    this.box = box && box.width > 0 ? [box.left, box.top, box.right, Math.max(box.bottom, hint?.height ? hint.bottom : 0)] : [-1e4, -1e4, -1e4, -1e4];
    const room = this.roomAt ? document.querySelector(this.roomAt)?.getBoundingClientRect() : undefined;
    if (room) {
      this.center = [room.left + room.width / 2, room.top + room.height * 0.5];
      this.flyW = Math.min(room.height * 0.8 * VIEW_W / VIEW_H, r.width * 0.92);
    } else {
      this.center = [column.left + column.width / 2, r.top + (top - r.top) * 0.47];
      this.flyW = Math.min(column.width * 0.86, (top - r.top) * 0.74 * VIEW_W / VIEW_H, r.width * 0.92);
    }
    this.lane = [column.left, column.right];
    if (!was) { this.fly = this.empty?.isConnected ? 1 : 0; this.wake(); }
    if (resized || !this.cpu) this.draw();
    else this.wake();
  }

  private buildAtlas(dpr: number, cw: number, ch: number) {
    const gl = this.gl;
    this.dpr = dpr;
    this.cellPx = [cw, ch];
    const cv = document.createElement("canvas");
    cv.width = ROW * cw;
    cv.height = 2 * ch;
    const ctx = cv.getContext("2d")!;
    const mono = getComputedStyle(document.documentElement).getPropertyValue("--mono").trim() || "monospace";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [row, set, weight] of [[0, SEA, 400], [1, TUBE, 700]] as const) {
      ctx.font = `${weight} ${FONT_PX * dpr}px ${mono}`;
      if (!ctx.font.includes(`${FONT_PX * dpr}px`)) ctx.font = `${weight} ${FONT_PX * dpr}px monospace`;
      [...set].forEach((g, i) => { if (g !== " ") ctx.fillText(g, (i + 0.5) * cw, (row + 0.5) * ch); });
    }
    if (this.cpu) {
      const px = ctx.getImageData(0, 0, cv.width, cv.height).data;
      this.cpu.ink = new Uint8ClampedArray(cv.width * cv.height);
      for (let i = 0; i < this.cpu.ink.length; i++) this.cpu.ink[i] = px[i * 4 + 3];
      this.cpu.inkW = cv.width;
      return;
    }
    if (!gl) return;
    if (!this.atlas) this.atlas = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  private buildCells(cols: number, rows: number) {
    const gl = this.gl;
    this.cols = cols;
    this.rows = rows;
    if (this.cpu) return void (this.cpu.cells = new Uint8ClampedArray(cols * rows * 4));
    if (!gl) return;
    if (!this.cellTex) this.cellTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.cellTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, cols, rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    if (!this.fbo) this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.cellTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private loc(prog: WebGLProgram, name: string) {
    const key = `${prog === this.cellsProg ? "c" : "g"}:${name}`;
    if (!this.u.has(key)) this.u.set(key, this.gl!.getUniformLocation(prog, name));
    return this.u.get(key)!;
  }

  private wake() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.live() || !(this.gl ? this.sdfTex : this.sdf)) { this.running = false; return; }
      const t = this.now(), blank = !!this.empty?.isConnected;
      const busy = this.moved > this.drawn || Math.abs(this.presence - this.pointer[2]) > 0.01 || this.fly !== (blank ? 1 : 0)
        || this.waves(t).length > 0;
      if (this.still.matches || t - this.drawn >= Math.max(busy ? 0 : 1 / (blank ? 24 : 12), this.cpu ? this.cost * (busy ? 2 : 4) : 0)) this.draw();
      if (this.still.matches) { this.running = false; return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  private draw() {
    const gl = this.gl;
    if (!this.sdf || this.seen[2] <= 0 || this.seen[3] <= 0 || (gl ? !this.cellTex || !this.atlas || !this.sdfTex : !this.cpu?.ink.length)) return;
    const t = this.now(), still = this.still.matches, [sx, sy, sw, sh] = this.seen;
    const blank = !!this.empty?.isConnected, dt = Math.min(0.1, t - this.drawn);
    this.fly = still ? Number(blank) : this.fly + (Number(blank) - this.fly) * (1 - Math.exp(-dt / (blank ? RISE : SINK)));
    if (Math.abs(Number(blank) - this.fly) < 0.003) this.fly = Number(blank);
    if (!blank) this.presence = 0;
    this.drawn = t;
    this.pointer[2] += (this.presence - this.pointer[2]) * (1 - Math.exp(-dt / HOVER));
    if (!gl) {
      const t0 = performance.now();
      this.cellsCpu(still ? 12 : t, still ? 0 : 1);
      this.glyphsCpu();
      this.cost = (performance.now() - t0) / 1000;
      return;
    }

    gl.disable(gl.BLEND);
    gl.enable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.cols, this.rows);
    gl.scissor(sx, this.rows - sy - sh, sw, sh);
    gl.useProgram(this.cellsProg);
    const c = (n: string) => this.loc(this.cellsProg, n);
    gl.uniform2f(c("uGrid"), this.cols, this.rows);
    gl.uniform2f(c("uCell"), this.cellPx[0] / this.dpr, this.cellPx[1] / this.dpr);
    gl.uniform4fv(c("uView"), this.view);
    gl.uniform4fv(c("uBox"), this.box);
    gl.uniform2f(c("uLane"), this.lane[0], this.lane[1]);
    gl.uniform1f(c("uFlyOn"), this.fly);
    gl.uniform1f(c("uTime"), still ? 12 : t);
    gl.uniform1f(c("uMotion"), still ? 0 : 1);
    const waves = this.waves(still ? 12 : t);
    waves.forEach((v, i) => this.packed.set([v.x, v.y, v.kind, v.amp, v.size, v.len], i * 8));
    gl.uniform1i(c("uWaves"), waves.length);
    gl.uniform4fv(c("uWave"), this.packed);
    const { half, bold } = strokes(this.flyW);
    gl.uniform3f(c("uFly"), this.flyW, half, bold);
    gl.uniform2f(c("uCenter"), this.center[0], this.center[1]);
    gl.uniform3f(c("uPointer"), this.pointer[0], this.pointer[1], still ? 0 : this.pointer[2]);
    gl.uniform2f(c("uSdfSize"), this.sdf.w, this.sdf.h);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sdfTex);
    gl.uniform1i(c("uSdf"), 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const [cw, ch] = this.cellPx;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.scissor(sx * cw, this.canvas.height - (sy + sh) * ch, sw * cw, sh * ch);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.glyphProg);
    const g = (n: string) => this.loc(this.glyphProg, n);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.cellTex);
    gl.uniform1i(g("uCells"), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.uniform1i(g("uAtlas"), 1);
    gl.uniform2f(g("uCellPx"), cw, ch);
    gl.uniform1f(g("uRows"), this.rows);
    gl.uniform1f(g("uHeight"), this.canvas.height);
    gl.uniform3fv(g("uSea"), this.colors.sea);
    gl.uniform3fv(g("uPink"), this.colors.pink);
    gl.uniform3fv(g("uHi"), this.colors.hi);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private cellsCpu(uTime: number, uMotion: number) {
    const cpu = this.cpu!, sdf = this.sdf!, out = cpu.cells, [sx, sy, sw, sh] = this.seen;
    const uCellX = this.cellPx[0] / this.dpr, uCellY = this.cellPx[1] / this.dpr, uFlyOn = this.fly;
    const uFlyW = this.flyW, { half: uFlyHalf, bold: uFlyBold } = strokes(this.flyW);
    const [cx0, cy0] = this.center, [vl, vt, vr, vb] = this.view, [bl, bt, br, bb] = this.box, [lane0, lane1] = this.lane;
    const [px0, py0, pz] = [this.pointer[0], this.pointer[1], uMotion ? this.pointer[2] : 0];
    const waves = this.waves(uTime);
    const flyAt = (x: number, y: number, c0: number, c1: number, w: number) => {
      const h = w * VIEW_H / VIEW_W, u = (x - c0) / w + 0.5, v = (y - c1) / h + 0.5;
      if (u < 0 || u > 1 || v < 0 || v > 1) return 1e4;
      const fx = u * sdf.w - 0.5, fy = v * sdf.h - 0.5, x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0;
      const at = (i: number, j: number) => sdf.data[clamp(j, 0, sdf.h - 1) * sdf.w + clamp(i, 0, sdf.w - 1)];
      const d = (at(x0, y0) * (1 - ax) + at(x0 + 1, y0) * ax) * (1 - ay) + (at(x0, y0 + 1) * (1 - ax) + at(x0 + 1, y0 + 1) * ax) * ay;
      return d * (w / sdf.w);
    };
    const shapeAt = (kind: number, dx: number, dy: number, r: number) => {
      if (kind < 0.5 || kind > 8.5) return Math.hypot(dx, dy) - r;
      if (kind < 1.5) return sdHeart(dx / r * 0.8, -dy / r * 0.8 + 0.55) * r / 0.8;
      if (kind < 2.5) return flyAt(dx, dy, 0, 0, r * 2.2);
      if (kind < 3.5) return sdStar5(dx / r, -dy / r, 1, 0.45) * r;
      return Math.hypot(dx, dy) - r * (0.78 + 0.22 * Math.cos(6 * Math.atan2(dy, dx)));
    };
    const height = (x: number, y: number) => {
      let sum = 0;
      for (const v of waves) {
        const d = shapeAt(v.kind, x - v.x, y - v.y, v.size) / v.len;
        sum += v.amp * Math.cos(6.2832 * d) * Math.exp(-0.9 * (d + 0.6) * (d + 0.6));
      }
      return sum;
    };
    const boxAt = (x: number, y: number) => {
      const qx = Math.abs(x - 0.5 * (bl + br)) - 0.5 * (br - bl) + 20, qy = Math.abs(y - 0.5 * (bt + bb)) - 0.5 * (bb - bt) + 20;
      return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - 20;
    };
    const t = uTime * uMotion, breathe = 1 + 0.018 * Math.sin(t * 0.9) * uMotion, sunk = smoothstep(0, 1, 1 - uFlyOn);
    const tube = TUBE.length, seaTop = SEA.length - 1;
    for (let cy = sy; cy < sy + sh; cy++) {
      for (let cx = sx; cx < sx + sw; cx++) {
        const px = (cx + 0.5) * uCellX, py = (cy + 0.5) * uCellY, o = (cy * this.cols + cx) * 4;
        const w = waves.length ? height(px, py) : 0;
        const slopeX = waves.length ? (height(px + 3, py) - w) / 3 : 0, slopeY = waves.length ? (height(px, py + 3) - w) / 3 : 0;
        const driftX = sunk ? sunk * 6 * snoise(px * 0.011, py * 0.011, t * 0.12) : 0, driftY = sunk ? sunk * 6 * snoise(px * 0.011 + 31, py * 0.011 + 31, t * 0.12) : 0;
        const qx = cx0 + (px - cx0) / breathe - slopeX * 30 + driftX, qy = cy0 + (py - cy0) - slopeY * 30 + driftY;
        const d = flyAt(qx, qy, cx0, cy0, uFlyW) - uFlyBold;
        const lift = Math.max(w, 0);
        if (d < 0.5 && uFlyOn > 0 && (uFlyOn >= 1 || 0.999 * (0.65 * (0.5 + 0.5 * snoise(px * 0.018, py * 0.018, 5))
          + 0.35 * fract(Math.sin(cx * 12.9898 + cy * 78.233) * 43758.5453)) < uFlyOn)) {
          const depth = clamp(-d / uFlyHalf, 0, 1), prof = Math.sqrt(Math.max(0, 1 - (1 - depth) * (1 - depth)));
          out[o] = ROW + Math.floor(clamp(1 + prof * (tube - 2) + 0.5 + lift * 0.5, 0, tube - 1));
          out[o + 1] = 255 * (0.58 + 0.42 * prof) * (0.4 + 0.6 * uFlyOn);
          out[o + 2] = 255;
          out[o + 3] = 255 * clamp(Math.max(0, prof - 0.55) * 1.4 * uFlyOn + lift * 0.3, 0, 1);
          continue;
        }
        const s = sea(px - slopeX * 90, py - slopeY * 90, t);
        const edge = smoothstep(0, 28, Math.min(Math.min(px - vl, vr - px), Math.min(py - vt, vb - py)));
        const hover = pz * Math.exp(-((px - px0) ** 2 + (py - py0) ** 2) / 3600);
        const ghost = 1 - smoothstep(-2, 6, d), gd = Math.max(d, 0) / 30, glow = Math.exp(-gd * gd) * uFlyOn;
        const calm = 1 + (0.3 + 0.7 * smoothstep(3, 40, d) - 1) * uFlyOn;
        const behind = 0.3 + 0.35 * smoothstep(0, 72, Math.max(lane0 - px, px - lane1));
        const quiet = (0.3 + 0.7 * smoothstep(0, 40, boxAt(px, py))) * (behind + (1 - behind) * uFlyOn);
        const v = clamp(s * calm + ghost * 0.3 + lift * 0.55 + Math.min(w, 0) * 0.15 + hover * 0.3, 0, 1);
        const bright = ((0.05 + 0.46 * v * v + 0.3 * glow + 0.3 * lift + 0.3 * hover) * quiet + 0.2 * ghost * Math.sqrt(quiet)) * edge;
        out[o] = Math.floor(clamp(v * 3.9, 0, seaTop));
        out[o + 1] = 255 * clamp(bright, 0, 1);
        out[o + 2] = 255 * clamp(glow * 0.85 + ghost * 0.7 + lift * 0.6 + hover * 0.5, 0, 1);
        out[o + 3] = 0;
      }
    }
  }

  private glyphsCpu() {
    const cpu = this.cpu!, [sx, sy, sw, sh] = this.seen, [cw, ch] = this.cellPx, W = sw * cw, H = sh * ch;
    if (!cpu.pixels || cpu.pixels.width !== W || cpu.pixels.height !== H) cpu.pixels = new ImageData(W, H);
    const px = cpu.pixels.data, { sea: c0, pink: c1, hi: c2 } = this.colors;
    px.fill(0);
    for (let cy = 0; cy < sh; cy++) {
      for (let cx = 0; cx < sw; cx++) {
        const o = ((cy + sy) * this.cols + cx + sx) * 4, g = cpu.cells[o];
        if (!g) continue;
        const b = cpu.cells[o + 2] / 255, a = cpu.cells[o + 3] / 255, bright = cpu.cells[o + 1] / 255;
        const r = 255 * ((c0[0] + (c1[0] - c0[0]) * b) * (1 - a) + c2[0] * a);
        const gg = 255 * ((c0[1] + (c1[1] - c0[1]) * b) * (1 - a) + c2[1] * a);
        const bb = 255 * ((c0[2] + (c1[2] - c0[2]) * b) * (1 - a) + c2[2] * a);
        const ax = (g % ROW) * cw, ay = Math.floor(g / ROW) * ch;
        for (let ly = 0; ly < ch; ly++) {
          let i = ((cy * ch + ly) * W + cx * cw) * 4, k = (ay + ly) * cpu.inkW + ax;
          for (let lx = 0; lx < cw; lx++, i += 4, k++) {
            const ink = cpu.ink[k];
            if (!ink) continue;
            px[i] = r; px[i + 1] = gg; px[i + 2] = bb; px[i + 3] = ink * bright;
          }
        }
      }
    }
    cpu.ctx.putImageData(cpu.pixels, sx * cw, sy * ch);
  }
}
