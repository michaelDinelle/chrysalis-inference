// A new chat's backdrop: the butterfly in an ocean of characters under the whole chat, the message box floating on it.
// The sea is a slow current (domain-warped noise over a swell), a click sends ripples across it in a shape (heart,
// ring, butterfly, star, flower) that bend the current as they pass, and the butterfly is drawn in the same
// characters, shaded like a lit tube, its light tinting the water near it. All of it is two WebGL2 passes: one fragment
// per character cell works out what that cell shows (pass 1, a cols x rows texture), then each screen pixel draws its
// cell's glyph from an atlas rendered at the screen's own pixel size (pass 2), so the CPU only sets uniforms. Without
// WebGL2 the page shows the logo as an image instead (attach() returns false).
import logo from "./logo-thin.svg";

// Glyph rows of the atlas: the sea's (light to dense, none that join into lines across a row) and the butterfly's in
// bold (edge to core). The orders follow measured ink in JetBrains Mono, so a gradient of values reads as an even
// gradient of glyphs.
const SEA = " ·~≈+*#";
const TUBE = "·:+*≈#%@";
const ROW = 32;
// The sea's cell, CSS px: small enough for the butterfly's strokes to be a few glyphs wide.
const CELL_W = 6, CELL_H = 11, FONT_PX = 10;
// The drawing's viewBox (logo.svg's), and its stroke half-width in those units (logo-thin.svg's strokes: 70% of
// logo.svg's 32.18).
const VIEW_W = 660, VIEW_H = 473, HALF_STROKE = 11.26;
// Click waves, in the order clicks cycle through them (the shader's kinds), each WAVE_LIFE s long; a drag leaves a wake
// (WAKE), small swells WAKE_LIFE s long.
const PATTERNS = [1, 0, 2, 3, 4];
const WAKE = 9, MAX_RIPPLES = 16, WAVE_LIFE = 5, WAKE_LIFE = 2;

const VERT = `#version 300 es
void main() { vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

// Pass 1. Simplex noise: Ian McEwan and Stefan Gustavson's (Ashima Arts, MIT). Shapes: Inigo Quilez's 2D distance
// functions (heart, five-point star), MIT.
const CELLS = `#version 300 es
precision highp float;
uniform vec2 uGrid, uCell;
uniform vec4 uView;         // the part of the sea on screen: left, top, right, bottom, CSS px from the window's corner
uniform vec4 uBox;          // the message box and the hint under it, likewise
uniform float uTime, uMotion;
uniform vec4 uRip[${MAX_RIPPLES}];
uniform vec2 uFly;          // the drawing's width and its strokes' half-width, CSS px
uniform vec2 uCenter;       // the butterfly's middle, CSS px
uniform sampler2D uSdf;     // distance to the drawing in texels, negative inside
uniform vec2 uSdfSize;
uniform vec3 uPointer;      // x, y (CSS px), presence 0..1
out vec4 o;

vec4 permute(vec4 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 1.0 / 7.0;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
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

// The current: noise pushed around by noise (domain warping) reads as slowly stirred water, and its ridges (where
// the noise crosses zero) are thin bright ribbons, like light on the floor of a pool. A swell rolls under it.
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

// Distance to the drawing, CSS px (negative inside its strokes), the drawing centered on c at width w.
float flyAt(vec2 p, vec2 c, float w) {
  float h = w * ${VIEW_H / VIEW_W};
  vec2 uv = (p - c) / vec2(w, h) + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1e4;
  return texture(uSdf, uv).r * (w / uSdfSize.x);
}

// A wave's shape: signed distance from its front, CSS px, the shape grown to size r around the click. The unit heart
// has its point at the origin and is about 1.1 tall: flipped upright, it is centered on the click, about 1.4 r wide.
float shapeAt(float kind, vec2 d, float r) {
  if (kind < 0.5 || kind > 8.5) return length(d) - r;
  if (kind < 1.5) return sdHeart(vec2(d.x, -d.y) / r * 0.8 + vec2(0.0, 0.55)) * r / 0.8;
  if (kind < 2.5) return flyAt(d, vec2(0.0), r * 2.2);
  if (kind < 3.5) return sdStar5(d * vec2(1.0, -1.0) / r, 1.0, 0.45) * r;
  float a = atan(d.y, d.x);
  return length(d) - r * (0.78 + 0.22 * cos(6.0 * a));
}

// A wave: a few smooth crests spreading from the click in its shape, quick at first and slowing as they spread (as a
// stone's rings do), lower as they go. A drag's wake is the same, smaller and shorter.
struct Wave { vec2 at; float kind, amp, size, len; };
bool wave(int i, out Wave v) {
  vec4 r = uRip[i];
  if (r.w < 0.0) return false;
  float age = uTime - r.z;
  bool wake = r.w > 8.5;
  float life = wake ? ${WAKE_LIFE}.0 : ${WAVE_LIFE}.0;
  if (age < 0.0 || age > life) return false;
  float k = 1.0 - age / life, size = 8.0 + (wake ? 90.0 : 320.0) * (1.0 - exp(-0.8 * age));
  v = Wave(r.xy, r.w, (wake ? 0.5 : 1.0) * pow(k, 1.6) * smoothstep(0.0, 0.3, age), size, wake ? 30.0 : 44.0 + 0.06 * size);
  return true;
}
// The water's height: each wave a short train of crests, at its front and behind it. A crest is a wavelength (len) of
// at least 44 px, so that it spans a few rows of 11 px cells: thinner, the grid missed it in most of them.
float waves(vec2 p) {
  float sum = 0.0;
  Wave v;
  for (int i = 0; i < ${MAX_RIPPLES}; i++) {
    if (!wave(i, v)) continue;
    float x = shapeAt(v.kind, p - v.at, v.size) / v.len;
    sum += v.amp * cos(6.2832 * x) * exp(-0.9 * (x + 0.6) * (x + 0.6));
  }
  return sum;
}

// Distance outside the message box (and the hint under it), CSS px.
float boxAt(vec2 p) {
  vec2 q = abs(p - 0.5 * (uBox.xy + uBox.zw)) - 0.5 * (uBox.zw - uBox.xy) + 20.0;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - 20.0;
}

void main() {
  vec2 cell = vec2(gl_FragCoord.x - 0.5, uGrid.y - 0.5 - gl_FragCoord.y);
  vec2 p = (cell + 0.5) * uCell;
  float t = uTime * uMotion;
  float w = waves(p);
  // what is seen through the water is bent by its slope: the current's ribbons, and the butterfly a little
  vec2 slope = vec2(waves(p + vec2(3.0, 0.0)) - w, waves(p + vec2(0.0, 3.0)) - w) / 3.0;
  float breathe = 1.0 + 0.018 * sin(t * 0.9) * uMotion;
  vec2 q = uCenter + (p - uCenter) / vec2(breathe, 1.0) - slope * 30.0;
  float d = flyAt(q, uCenter, uFly.x);
  float edge = smoothstep(0.0, 28.0, min(min(p.x - uView.x, uView.z - p.x), min(p.y - uView.y, uView.w - p.y)));
  float hover = uPointer.z * exp(-dot2(p - uPointer.xy) / 3600.0);
  float lift = max(w, 0.0);
  if (d < 0.5) {
    float depth = clamp(-d / uFly.y, 0.0, 1.0);
    float prof = sqrt(max(0.0, 1.0 - (1.0 - depth) * (1.0 - depth)));
    float g = ${ROW}.0 + floor(clamp(1.0 + prof * ${TUBE.length - 2}.0 + 0.5 + lift * 0.5, 0.0, ${TUBE.length - 1}.0));
    o = vec4(g / 255.0, 0.58 + 0.42 * prof, 1.0, clamp(max(0.0, prof - 0.55) * 1.4 + lift * 0.3, 0.0, 1.0));
    return;
  }
  float s = sea(p - slope * 90.0, t);
  float gd = max(d, 0.0) / 30.0;
  float glow = exp(-gd * gd);
  // the water right around the butterfly lies still, so its outline stays clean (its light still tints it), and the
  // water around the message box is dimmer, so the box and the hint under it read as resting on it
  float calm = mix(0.3, 1.0, smoothstep(3.0, 40.0, d));
  float quiet = mix(0.3, 1.0, smoothstep(0.0, 40.0, boxAt(p)));
  float v = clamp(s * calm + lift * 0.55 + min(w, 0.0) * 0.15 + hover * 0.3, 0.0, 1.0);
  float g = floor(clamp(v * 3.9, 0.0, ${SEA.length - 1}.0));
  float bright = (0.05 + 0.46 * v * v + 0.3 * glow + 0.3 * lift + 0.3 * hover) * edge * quiet;
  o = vec4(g / 255.0, clamp(bright, 0.0, 1.0), clamp(glow * 0.85 + lift * 0.6 + hover * 0.5, 0.0, 1.0), 0.0);
}`;

// Pass 2: each screen pixel draws its cell's glyph.
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

// Squared distances to the nearest zero of f, in place along one line (Felzenszwalb and Huttenlocher's transform).
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

// The drawing as a signed distance field: texels from its nearest edge, negative inside its strokes.
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
  const m = css.trim().match(/^#([0-9a-f]{6})$/i);
  const n = m ? parseInt(m[1], 16) : 0x888888;
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
};

export class Ocean {
  readonly canvas = document.createElement("canvas");
  private gl: WebGL2RenderingContext | null = null;
  private cellsProg!: WebGLProgram;
  private glyphProg!: WebGLProgram;
  private u = new Map<string, WebGLUniformLocation | null>();
  private cellTex: WebGLTexture | null = null;
  private fbo: WebGLFramebuffer | null = null;
  private atlas: WebGLTexture | null = null;
  private sdfTex: WebGLTexture | null = null;
  private sdf: Sdf | null = null;
  private chat: HTMLElement | null = null;
  private cols = 0;
  private rows = 0;
  private dpr = 0;
  private cellPx = [0, 0];
  private view = [0, 0, 0, 0];
  private seen = [0, 0, 0, 0];                      // the cells the layer shows: column, row, columns, rows
  private box = [-1e4, -1e4, -1e4, -1e4];
  private center = [0, 0];
  private flyW = 0;
  private ripples = new Float32Array(MAX_RIPPLES * 4).fill(-1);
  private next = 0;
  private pattern = 0;
  private t0 = performance.now() / 1000;
  private pointer = [0, 0, 0];
  private presence = 0;
  private lastWake: [number, number] | null = null;
  private colors = { sea: [0, 0, 0], pink: [0, 0, 0], hi: [0, 0, 0] };
  private drawn = 0;
  private running = false;
  private still = matchMedia("(prefers-reduced-motion: reduce)");
  private dark = matchMedia("(prefers-color-scheme: dark)");
  private resize = new ResizeObserver(() => this.layout());

  // `layer`: the room the sea fills (under the top bar, behind the message box), shown while a new chat is.
  constructor(private layer: HTMLElement) {
    const c = this.canvas;
    c.className = "ocean";
    c.setAttribute("aria-hidden", "true");
    // a click on the water keeps the message box's focus: typing goes on where it was. The canvas sits at the
    // window's corner, so client coordinates are the shader's.
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      this.splash(e.clientX, e.clientY, PATTERNS[this.pattern++ % PATTERNS.length]);
      this.lastWake = [e.clientX, e.clientY];
    });
    c.addEventListener("pointermove", (e) => {
      this.pointer[0] = e.clientX; this.pointer[1] = e.clientY; this.presence = 1;
      if (e.buttons & 1 && this.lastWake && Math.hypot(e.clientX - this.lastWake[0], e.clientY - this.lastWake[1]) > 34) {
        this.splash(e.clientX, e.clientY, WAKE);
        this.lastWake = [e.clientX, e.clientY];
      }
      this.wake();
    });
    c.addEventListener("pointerleave", () => { this.presence = 0; this.lastWake = null; });
    c.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.gl = null; this.running = false; });
    c.addEventListener("webglcontextrestored", () => { if (this.init()) this.layout(); });
    this.dark.addEventListener("change", () => { this.readColors(); this.wake(); });
    this.still.addEventListener("change", () => this.wake());
  }

  private init(): boolean {
    const gl = this.canvas.getContext("webgl2", { premultipliedAlpha: true, antialias: false, alpha: true });
    if (!gl) return false;
    this.gl = gl;
    try {
      this.cellsProg = compile(gl, CELLS);
      this.glyphProg = compile(gl, GLYPHS);
    } catch (e) {
      console.error(e);
      this.gl = null;
      return false;
    }
    this.u.clear();
    this.cellTex = this.fbo = this.atlas = this.sdfTex = null;
    this.cols = this.rows = this.dpr = 0;
    if (this.sdf) this.uploadSdf();
    return true;
  }

  // Runs the sea while `chat` (a new chat's empty state) is in the page; false without WebGL2, for the caller to show
  // the plain logo.
  attach(chat: HTMLElement): boolean {
    if (!this.gl && !this.init()) return false;
    this.chat = chat;
    if (!this.canvas.isConnected) {
      this.layer.append(this.canvas);
      this.resize.observe(this.layer);
      const box = document.querySelector(".composer-box");
      if (box) this.resize.observe(box);
    }
    this.readColors();
    if (!this.sdf) void drawingSdf(720).then((s) => { this.sdf = s; this.uploadSdf(); this.wake(); });
    this.wake();
    return true;
  }

  private live() { return !!this.gl && !!this.chat?.isConnected; }
  private now() { return performance.now() / 1000 - this.t0; }

  // The theme's colors: the sea in --muted, the butterfly in --accent, its light lighter in the dark, deeper in light.
  private readColors() {
    const style = getComputedStyle(document.documentElement);
    this.colors = { sea: rgb(style.getPropertyValue("--muted")), pink: rgb(style.getPropertyValue("--accent")),
      hi: this.dark.matches ? [1.0, 0.86, 0.92] : rgb(style.getPropertyValue("--accent-hi")) };
  }

  private splash(x: number, y: number, kind: number) {
    if (this.still.matches) return;
    const i = this.next++ % MAX_RIPPLES;
    this.ripples.set([x, y, this.now(), kind], i * 4);
    this.wake();
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

  // The grid covers the window from its corner, whole device pixels per cell, and the layer shows the part of it under
  // the chat. So a side panel sliding open or shut only moves the layer's edge: every character stays where it is on
  // screen, and the canvas keeps its size (resizing it would blank it until the next frame is drawn: a flicker on
  // every frame of the slide). The frame is drawn here, before the browser paints the new layout.
  private layout() {
    const gl = this.gl;
    if (!gl || !this.live()) return;
    const r = this.layer.getBoundingClientRect();
    if (r.width < 40 || r.height < 40) return;
    const dpr = devicePixelRatio || 1, cw = Math.max(1, Math.round(CELL_W * dpr)), ch = Math.max(1, Math.round(CELL_H * dpr));
    const cols = Math.ceil(innerWidth * dpr / cw), rows = Math.ceil(innerHeight * dpr / ch);
    if (dpr !== this.dpr || cw !== this.cellPx[0] || ch !== this.cellPx[1]) this.buildAtlas(dpr, cw, ch);
    if (cols !== this.cols || rows !== this.rows || !this.cellTex) {
      this.buildCells(cols, rows);
      this.canvas.width = cols * cw;
      this.canvas.height = rows * ch;
      Object.assign(this.canvas.style, { width: `${cols * cw / dpr}px`, height: `${rows * ch / dpr}px` });
    }
    Object.assign(this.canvas.style, { left: `${-r.left}px`, top: `${-r.top}px` });
    this.view = [r.left, r.top, r.right, r.bottom];
    const x0 = Math.max(0, Math.floor(r.left * dpr / cw)), y0 = Math.max(0, Math.floor(r.top * dpr / ch));
    this.seen = [x0, y0, Math.min(cols, Math.ceil(r.right * dpr / cw)) - x0, Math.min(rows, Math.ceil(r.bottom * dpr / ch)) - y0];
    // the butterfly over the message box, in the middle of the room above it
    const box = document.querySelector(".composer-box")?.getBoundingClientRect();
    const hint = document.querySelector("#composer .hint")?.getBoundingClientRect();
    const column = box && box.width > 0 ? box : r, top = box && box.width > 0 ? box.top : r.bottom;
    this.box = box && box.width > 0 ? [box.left, box.top, box.right, Math.max(box.bottom, hint?.height ? hint.bottom : 0)] : [-1e4, -1e4, -1e4, -1e4];
    this.center = [column.left + column.width / 2, r.top + (top - r.top) * 0.47];
    this.flyW = Math.min(column.width * 0.86, (top - r.top) * 0.74 * VIEW_W / VIEW_H, r.width * 0.92);
    this.draw();
  }

  private buildAtlas(dpr: number, cw: number, ch: number) {
    const gl = this.gl!;
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
      // a font string the canvas cannot parse is ignored without a word (leaving 10px sans-serif): monospace then
      ctx.font = `${weight} ${FONT_PX * dpr}px ${mono}`;
      if (!ctx.font.includes(`${FONT_PX * dpr}px`)) ctx.font = `${weight} ${FONT_PX * dpr}px monospace`;
      [...set].forEach((g, i) => { if (g !== " ") ctx.fillText(g, (i + 0.5) * cw, (row + 0.5) * ch); });
    }
    if (!this.atlas) this.atlas = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.atlas);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  private buildCells(cols: number, rows: number) {
    const gl = this.gl!;
    this.cols = cols;
    this.rows = rows;
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

  // Runs the loop while the sea is on screen: every frame while something moves fast (a wave, the pointer's light),
  // 24 a second for the slow current alone (on a 144 Hz screen every frame would be six times the work for nothing to
  // see), once with reduced motion. requestAnimationFrame stops by itself in a hidden tab.
  private wake() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.live() || !this.sdfTex) { this.running = false; return; }
      const t = this.now();
      const busy = this.presence > 0 || this.pointer[2] > 0.01 || this.ripples.some((v, i) => i % 4 === 2 && v >= 0 && t - v < WAVE_LIFE);
      if (busy || t - this.drawn >= 1 / 24) this.draw();
      if (this.still.matches) { this.running = false; return; }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // Both passes only over the cells the layer shows: the rest of the window's grid is under the side panels.
  private draw() {
    const gl = this.gl;
    if (!gl || !this.cellTex || !this.atlas || !this.sdf || !this.sdfTex || this.seen[2] <= 0 || this.seen[3] <= 0) return;
    const t = this.now(), still = this.still.matches, [sx, sy, sw, sh] = this.seen;
    this.drawn = t;
    this.pointer[2] += (this.presence - this.pointer[2]) * 0.12;

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
    gl.uniform1f(c("uTime"), still ? 12 : t);
    gl.uniform1f(c("uMotion"), still ? 0 : 1);
    gl.uniform4fv(c("uRip"), this.ripples);
    gl.uniform2f(c("uFly"), this.flyW, HALF_STROKE * this.flyW / VIEW_W);
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
}
