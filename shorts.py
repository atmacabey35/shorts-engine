#!/usr/bin/env python3
"""
Shorts engine - 3 code-drawn engines, no AI video, no paid API, no internet.

    python shorts.py auto       # 5-agent pipeline: next unused "How many X fit inside Y?" (reads data/bodies.json + ledger.json)
    python shorts.py scale      # same as auto (--seed N picks a ranked concept without touching the ledger)
    python shorts.py illusion   # optical illusions with a reveal
    python shorts.py physics    # "Which colour survives?" survival race
    python shorts.py all        # one video from each engine

Options:  --seed N   pick a specific variant      --out DIR   output folder (default: out)
Needs: Python 3, numpy, Pillow, ffmpeg + ffprobe.
Every video gets a matching .txt (title / description / tags) and is checked
automatically (resolution, codec, duration, audio, not-black frames).
"""
import sys, os, math, json, shutil, subprocess, tempfile, wave, argparse, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 44100
CY = 1000
OUTDIR = "out"


def fail(msg, code=1):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(code)


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


# =============================================================== text
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/system/fonts/Roboto-Bold.ttf",
    "/system/fonts/DroidSans-Bold.ttf",
    "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]
_fc = {}


def font(size):
    if size in _fc:
        return _fc[size]
    f = None
    for p in FONT_PATHS:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                break
            except OSError:
                pass
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:
            f = ImageFont.load_default()
    _fc[size] = f
    return f


_m = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def tlen(text, f):
    return _m.textlength(text, font=f)


def fit_font(text, max_w, size0, min_size=24):
    for size in range(size0, min_size - 1, -2):
        if tlen(text, font(size)) <= max_w:
            return font(size)
    return font(min_size)


def wrap_fit(text, max_w, max_lines, size0):
    for size in range(size0, 29, -2):
        f = font(size)
        lines, cur = [], ""
        for w in text.split():
            t = (cur + " " + w).strip()
            if tlen(t, f) <= max_w:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        if len(lines) <= max_lines and all(tlen(l, f) <= max_w for l in lines):
            return f, lines
    return font(30), [text]


class Text:
    def __init__(self):
        self.img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def _draw(self, xy, text, f, fill):
        try:
            self.d.text(xy, text, font=f, fill=fill, anchor="mm")
        except (ValueError, TypeError):
            w = tlen(text, f)
            self.d.text((xy[0] - w / 2, xy[1] - 20), text, font=f, fill=fill)

    def put(self, xy, text, f, col=(255, 255, 255), a=1.0, shadow=True):
        if a <= 0.01:
            return
        al = int(255 * min(a, 1.0))
        if shadow:
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (3, 4), (-2, 2)):
                self._draw((xy[0] + dx, xy[1] + dy), text, f, (0, 0, 0, int(al * 0.85)))
        self._draw(xy, text, f, (col[0], col[1], col[2], al))

    def lines(self, cx, cy, lines, f, col=(255, 255, 255), a=1.0, lh=None):
        lh = lh or int(getattr(f, "size", 60) * 1.18)
        y0 = cy - (len(lines) - 1) * lh / 2
        for i, l in enumerate(lines):
            self.put((cx, y0 + i * lh), l, f, col, a)

    def apply(self, rgb_u8):
        base = Image.fromarray(rgb_u8).convert("RGBA")
        return Image.alpha_composite(base, self.img).convert("RGB")


# =============================================================== background / glow
def make_bg(top, bot, seed=7, stars=320, vignette=0.55):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = (np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t) * np.ones((1, W, 1), np.float32)
    for _ in range(stars):
        sx, sy = int(rng.integers(0, W - 2)), int(rng.integers(0, H - 2))
        v = float(rng.integers(90, 230))
        s = 2 if rng.random() < 0.2 else 1
        bg[sy:sy + s, sx:sx + s, :] = np.maximum(bg[sy:sy + s, sx:sx + s, :], v)
    if vignette:
        yy = (np.arange(H, dtype=np.float32)[:, None] - H / 2) / (H / 2)
        xx = (np.arange(W, dtype=np.float32)[None, :] - W / 2) / (W / 2)
        v = 1 - vignette * np.clip((xx * xx * 0.65 + yy * yy * 0.55) - 0.2, 0, 1)
        bg *= v[..., None]
    return bg


def bloom(frame, strength=0.55, thresh=165.0, radius=7):
    small = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BOX)
    arr = np.clip(np.asarray(small, np.float32) - thresh, 0, 255) * (255.0 / (255.0 - thresh))
    g = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)).resize((W, H), Image.BILINEAR)
    return frame + np.asarray(g, np.float32) * strength


X = np.arange(W, dtype=np.float32)[None, :]
Y = np.arange(H, dtype=np.float32)[:, None]


def region(cx, cy, ext):
    x0, x1 = max(0, int(cx - ext)), min(W, int(cx + ext) + 2)
    y0, y1 = max(0, int(cy - ext)), min(H, int(cy + ext) + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, x1, y0, y1


def blend(frame, reg, col, a):
    x0, x1, y0, y1 = reg
    sub = frame[y0:y1, x0:x1]
    a3 = a[..., None]
    sub *= (1.0 - a3)
    sub += col * a3


def disc(frame, cx, cy, r, col, alpha=1.0, glossy=True):
    if alpha <= 0.01 or r < 0.5:
        return
    reg = region(cx, cy, r + 3)
    if reg is None:
        return
    x0, x1, y0, y1 = reg
    dx = X[:, x0:x1] - np.float32(cx)
    dy = Y[y0:y1, :] - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy)
    a = np.clip(r - d + 0.5, 0, 1) * np.float32(alpha)
    c = np.array(col, np.float32)
    if glossy:
        light = np.clip(0.72 + 0.4 * (-(dx + dy) / (1.414 * r)), 0.45, 1.25)
        hl = np.clip(1 - np.sqrt((dx + 0.35 * r) ** 2 + (dy + 0.38 * r) ** 2) / (0.42 * r), 0, 1) ** 2
        colr = c * light[..., None] + 120 * hl[..., None]
    else:
        colr = c
    blend(frame, reg, colr, a)


# =============================================================== audio
class Audio:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.sig = np.zeros(self.n)

    def add(self, t, snd, gain=1.0):
        i = int(t * SR)
        if i < 0 or i >= self.n:
            return
        j = min(self.n, i + len(snd))
        self.sig[i:j] += snd[:j - i] * gain

    def save(self, path):
        s = self.sig.copy()
        s *= np.clip((self.n - np.arange(self.n)) / (0.15 * SR), 0, 1)
        s = s / max(np.max(np.abs(s)), 1e-9) * 0.8
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes((s * 32767).astype("<i2").tobytes())


def whoosh(dur, f0, f1, amp=0.5, seed=1):
    n = int(dur * SR)
    noise = np.random.default_rng(seed).standard_normal(n)
    out = np.zeros(n)
    y = 0.0
    for i in range(n):
        f = f0 + (f1 - f0) * (i / n)
        a = 1.0 - math.exp(-2 * math.pi * f / SR)
        y += a * (noise[i] - y)
        out[i] = y
    out /= (np.max(np.abs(out)) + 1e-9)
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 2 * amp


def tick(freq=1300.0, amp=0.28):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.012) * amp


def boom(dur=0.9, amp=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (70 - 25 * t / dur) * t) * np.exp(-t / 0.28) * amp


def bell(freq, dur=0.5, amp=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * freq * 2.01 * t)
         + 0.15 * np.sin(2 * np.pi * freq * 3.02 * t))
    return s * np.exp(-t / (dur / 4.5)) * amp * (1 - np.exp(-t / 0.002))


def pop(amp=0.4):
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (500 - 1800 * t) * t) * np.exp(-t / 0.05) * amp


_grid = np.linspace(0, 1, 2001)
_sm = _grid * _grid * (3 - 2 * _grid)


def sm_inv(y):
    return float(np.interp(y, _sm, _grid))


PENTA = [0, 2, 4, 7, 9, 12, 14, 16]


def note(k, base=261.63):
    return base * 2 ** (PENTA[k % len(PENTA)] / 12.0)


# =============================================================== writer + QC + metadata
def render_video(path, dur, frame_fn, audio, crf=17):
    nf = int(round(dur * FPS))
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        audio.save(wav)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-i", wav,
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-t", str(dur), "-movflags", "+faststart", path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(nf):
                p.stdin.write(frame_fn(i / FPS))
                if i % 90 == 0:
                    print("  frame %d/%d" % (i, nf), flush=True)
            p.stdin.close()
        except BrokenPipeError:
            fail("ffmpeg closed the pipe early", 6)
        if p.wait() != 0:
            fail("ffmpeg failed", 6)


def quality_check(path, dur):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("ffprobe failed: " + r.stderr, 4)
    info = json.loads(r.stdout)
    vs = [s for s in info["streams"] if s["codec_type"] == "video"]
    au = [s for s in info["streams"] if s["codec_type"] == "audio"]
    bad = []
    if len(vs) != 1:
        bad.append("video streams: %d" % len(vs))
    else:
        v = vs[0]
        if (v["width"], v["height"]) != (W, H):
            bad.append("resolution %sx%s" % (v["width"], v["height"]))
        if v.get("codec_name") != "h264":
            bad.append("codec " + str(v.get("codec_name")))
        if v.get("pix_fmt") != "yuv420p":
            bad.append("pix_fmt " + str(v.get("pix_fmt")))
        n, d = v["r_frame_rate"].split("/")
        if abs(float(n) / float(d) - FPS) > 0.01:
            bad.append("fps " + v["r_frame_rate"])
    if not au:
        bad.append("no audio")
    got = float(info["format"]["duration"])
    if abs(got - dur) > 0.3:
        bad.append("duration %.2f" % got)
    if not bad:
        for t in (dur * 0.1, dur * 0.35, dur * 0.6, dur * 0.85):
            raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "%.2f" % t, "-i", path, "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
            if len(raw) != W * H * 3:
                bad.append("cannot read frame at %.1fs" % t)
                continue
            a = np.frombuffer(raw, np.uint8)
            if a.mean() < 8 or a.std() < 8:
                bad.append("frame at %.1fs is blank/flat" % t)
    if bad:
        fail("quality check failed: " + "; ".join(bad), 5)
    print("QC OK: %dx%d h264/yuv420p %dfps, audio, %.2fs, frames not blank" % (W, H, FPS, got))


def write_meta(path_txt, title, desc, tags, notes=""):
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("TITLE:\n%s\n\nDESCRIPTION:\n%s\n\nTAGS:\n%s\n" % (title, desc, ", ".join(tags)))
        if notes:
            f.write("\nNOTES:\n%s\n" % notes)


# =============================================================== ENGINE 1: SCALE
DEFAULT_RAW = {
    "moon": {"name": "The Moon", "plural": "Moons", "diam_km": 3474.8, "color_hex": "#B0B0AC"},
    "mercury": {"name": "Mercury", "plural": "Mercuries", "diam_km": 4879.4, "color_hex": "#A08E7E"},
    "mars": {"name": "Mars", "plural": "Mars-sized worlds", "diam_km": 6779.0, "color_hex": "#C4623E"},
    "earth": {"name": "Earth", "plural": "Earths", "diam_km": 12742.0, "color_hex": "#2364D7"},
    "neptune": {"name": "Neptune", "plural": "Neptunes", "diam_km": 49244.0, "color_hex": "#3C64D7"},
    "saturn": {"name": "Saturn", "plural": "Saturns", "diam_km": 116460.0, "color_hex": "#CEB078"},
    "jupiter": {"name": "Jupiter", "plural": "Jupiters", "diam_km": 139820.0, "color_hex": "#B0764E"},
    "sun": {"name": "The Sun", "plural": "Suns", "diam_km": 1392700.0, "color_hex": "#FFBD3C"},
}
BODIES = {}
MIN_RATIO, MAX_RATIO = 8.0, 120.0


def _hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lighten(c, k):
    return tuple(int(min(255, x + (255 - x) * k)) for x in c)


def normalize_body(key, raw):
    d = float(raw["diam_km"])
    col = _hex(raw.get("color_hex", "#999999"))
    t = raw.get("type")
    if t is None:
        t = "earth" if key == "earth" else ("star" if d >= 2.0e5 else ("gas" if d >= 2.0e4 else "rock"))
    style = {"star": "sun", "gas": "gas", "ice": "gas", "rock": "rock", "earth": "earth"}[t]
    return dict(key=key, name=raw["name"], plural=raw.get("plural", raw["name"] + "s"), diam=d, vol=raw.get("vol_km3"),
                style=style, col=col, col2=tuple(raw["col2"]) if "col2" in raw else _lighten(col, 0.45),
                bands=int(raw.get("bands", 8 + int(d) % 9)), spot=bool(raw.get("spot", key == "jupiter")),
                fam=float(raw.get("familiarity", 3)), core=_lighten(col, 0.55),
                edge=(int(col[0] * 0.96), int(col[1] * 0.65), int(col[2] * 0.35)))


def load_bodies(path):
    raw = DEFAULT_RAW
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)["bodies"]
    BODIES.clear()
    for k, v in raw.items():
        BODIES[k] = normalize_body(k, v)
    return BODIES


def nm_mid(n):
    return ("the " + n[4:]) if n.startswith("The ") else n


def nm_cap(n):
    return n[0].upper() + n[1:]


# ---- procedural 3D sphere shader with enhanced atmospheric rim
_nr = np.random.default_rng(42)
_NW = _nr.normal(size=(8, 3)).astype(np.float32)
_NW /= np.linalg.norm(_NW, axis=1, keepdims=True)
_NW2 = _nr.normal(size=(8, 3)).astype(np.float32)
_NW2 /= np.linalg.norm(_NW2, axis=1, keepdims=True)
_NP = _nr.uniform(0, 2 * np.pi, 8).astype(np.float32)
LIGHT = np.array([-0.55, -0.45, 0.70], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def fbm(px, py, pz, freq=1.0, octaves=4, off=0.0):
    tot = 0.0
    amp = 1.0
    norm = 0.0
    for k in range(octaves):
        f = freq * (1.9 ** k)
        s1 = (px * _NW[k, 0] + py * _NW[k, 1] + pz * _NW[k, 2]) * f + _NP[k] + off
        s2 = (px * _NW2[k, 0] + py * _NW2[k, 1] + pz * _NW2[k, 2]) * f * 0.8 + _NP[(k + 3) % 8]
        tot = tot + amp * np.sin(s1 + 1.1 * np.sin(s2))
        norm += amp
        amp *= 0.55
    return tot / norm


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def body_rgb(key, dx, dy, r, spin=0.0, t=0.0):
    b = BODIES[key]
    u, v = np.broadcast_arrays(dx / r, dy / r)
    rr = np.clip(u * u + v * v, 0, 1)
    z = np.sqrt(1 - rr)
    c, s_ = math.cos(spin), math.sin(spin)
    px, py, pz = u * c + z * s_, v, -u * s_ + z * c
    diff = np.clip(u * LIGHT[0] + v * LIGHT[1] + z * LIGHT[2], 0, 1)
    lit = 0.08 + 0.92 * (diff ** 0.80)
    st = b["style"]
    base = np.array(b["col"], np.float32)

    if st == "earth":
        h = fbm(px, py, pz, 1.5, 5)
        land = sstep(0.12, 0.2, h)
        shallow = sstep(-0.1, 0.14, h)
        ocean = np.array([6, 35, 110], np.float32) * (1 - shallow[..., None]) + np.array([25, 115, 195], np.float32) * shallow[..., None]
        lat = np.abs(py)
        veg = sstep(0.0, 0.5, fbm(pz, px, py, 2.3, 3) * 0.5 + 0.5 - lat * 0.6)
        ground = np.array([145, 120, 75], np.float32) * (1 - veg[..., None]) + np.array([45, 130, 55], np.float32) * veg[..., None]
        col = ocean * (1 - land[..., None]) + ground * land[..., None]
        ice = sstep(0.84, 0.9, lat + 0.05 * h)
        col = col * (1 - ice[..., None]) + np.array([240, 245, 250], np.float32) * ice[..., None]
        cl = sstep(0.38, 0.62, fbm(px, py, pz, 2.6, 4, off=t * 0.6)) * 0.9
        col = col * (1 - 0.8 * cl[..., None]) + 255 * 0.8 * cl[..., None]
        refl = 2 * diff * z - LIGHT[2]
        spec = np.clip(refl, 0, 1) ** 40 * (1 - land) * (1 - cl) * 0.65 * (diff > 0)
        out = col * lit[..., None] + 255 * spec[..., None]
        rim = (1 - z) ** 2.5 * (0.35 + 0.65 * diff)
        out = out + np.array([90, 160, 255], np.float32) * rim[..., None] * 1.15
        return out

    if st == "gas":
        turb = fbm(px, py, pz, 1.4, 4)
        band = np.sin(py * b["bands"] + 1.5 * turb) * 0.5 + 0.5
        c2 = np.array(b["col2"], np.float32)
        col = base * (1 - band[..., None]) + c2 * band[..., None]
        if b.get("spot"):
            dd = np.sqrt((px - 0.55) ** 2 + (py - 0.28) ** 2 + (pz - 0.78) ** 2)
            sp = sstep(0.32, 0.14, dd)
            col = col * (1 - sp[..., None]) + np.array([195, 75, 45], np.float32) * sp[..., None]
        out = col * lit[..., None]
        rim = (1 - z) ** 2.2 * (0.3 + 0.7 * diff)
        return out + np.array(b["col2"], np.float32) * rim[..., None] * 0.65

    if st == "sun":
        gran = fbm(px, py, pz, 7.0, 4, off=t * 0.5)
        spots = sstep(0.55, 0.8, fbm(px, py, pz, 1.2, 3, off=3.0))
        lim = np.clip(1 - 0.55 * (1 - z) ** 1.2, 0.35, 1)
        core, edge = np.array(b["core"], np.float32), np.array(b["edge"], np.float32)
        tt = (1 - z) ** 0.85
        col = core * (1 - tt[..., None]) + edge * tt[..., None]
        k = (0.94 + 0.10 * gran) * (1 - 0.45 * spots) * lim
        return col * k[..., None]

    # rock
    h = fbm(px, py, pz, 2.2, 5)
    alb = 0.70 + 0.30 * h
    cr = np.sin(px * 7.5) * np.sin(py * 7.5 + 1.0) * np.sin(pz * 7.5 - 0.5)
    crater = sstep(0.35, 0.7, cr)
    rimc = sstep(0.2, 0.35, cr) * (1 - sstep(0.35, 0.5, cr))
    col = base * (alb * (1 - 0.35 * crater) + 0.22 * rimc)[..., None]
    rim = (1 - z) ** 3.0 * (0.2 + 0.8 * diff)
    return col * lit[..., None] + np.array(b["col"], np.float32) * rim[..., None] * 0.25


def draw_saturn_rings(frame, cx, cy, r, alpha=1.0, behind=False):
    """Draws procedural Saturn rings behind or in front of the sphere."""
    reg = region(cx, cy, r * 2.5)
    if reg is None:
        return
    x0, x1, y0, y1 = reg
    dx = (X[:, x0:x1] - np.float32(cx))
    dy = (Y[y0:y1, :] - np.float32(cy))
    tilt_y = dy * 2.6
    dist = np.sqrt(dx * dx + tilt_y * tilt_y)
    r_in, r_out = r * 1.30, r * 2.25
    in_ring = (dist >= r_in) & (dist <= r_out)
    if behind:
        in_ring = in_ring & (dy < 0)
    else:
        in_ring = in_ring & (dy >= 0)
    if not np.any(in_ring):
        return
    r_norm = np.clip((dist - r_in) / (r_out - r_in), 0, 1)
    gap = np.clip(1.0 - np.abs(r_norm - 0.65) * 14.0, 0, 1)
    tex = (0.75 + 0.25 * np.sin(r_norm * 45.0)) * (1.0 - 0.85 * gap)
    ring_col = np.array([215, 195, 145], np.float32) * tex[..., None]
    ring_a = in_ring.astype(np.float32) * 0.78 * alpha
    blend(frame, reg, ring_col, ring_a)


def draw_body(frame, key, cx, cy, r, alpha=1.0, darken=0.0, spin=0.0, t=0.0):
    if alpha <= 0.01:
        return
    sun = BODIES[key]["style"] == "sun"
    is_saturn = (key == "saturn")
    ext = r * 1.8 if sun else (r * 2.4 if is_saturn else r * 1.25)
    r = max(r, 2.4)
    reg = region(cx, cy, ext)
    if reg is None:
        return
    x0, x1, y0, y1 = reg
    dx = X[:, x0:x1] - np.float32(cx)
    dy = Y[y0:y1, :] - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy)
    inside = np.clip(r - d + 0.5, 0, 1)

    # 1. Atmospheric Outer Glow
    glow_col = np.array(BODIES[key]["col2"] if not sun else BODIES[key]["edge"], np.float32)
    glow_dist = d - r
    glow_a = np.where(glow_dist > 0, np.exp(-glow_dist / (0.16 * r)), 0.0).astype(np.float32)
    glow_intensity = 0.65 if not sun else 0.85
    blend(frame, reg, glow_col, glow_a * glow_intensity * alpha * (1.0 - 0.4 * darken))

    # 2. Saturn rings (behind sphere)
    if is_saturn:
        draw_saturn_rings(frame, cx, cy, r, alpha, behind=True)

    # 3. Sphere shading
    h_, w_ = d.shape
    half = h_ * w_ > 600000

    def shade(dxs, dys):
        return body_rgb(key, dxs, dys, r, spin, t)

    if half:
        col = shade(dx[:, ::2], dy[::2, :])
        col = np.repeat(np.repeat(col, 2, axis=0), 2, axis=1)[:h_, :w_]
    else:
        col = shade(dx, dy)

    if darken > 0:
        col = col * np.float32(1.0 - 0.45 * darken)

    blend(frame, reg, col, inside * alpha)

    # 4. Saturn rings (front of sphere)
    if is_saturn:
        draw_saturn_rings(frame, cx, cy, r, alpha, behind=False)


class DotLayer:
    def __init__(self, inner, ratio, outer_px_r=470.0, S=1000):
        self.S = S
        dd = 2 * outer_px_r / ratio
        pitch, rad = dd * 1.04, dd * 0.5 * 0.93
        pts = []
        y, row = -outer_px_r, 0
        while y <= outer_px_r:
            x = -outer_px_r + (pitch / 2 if row % 2 else 0)
            while x <= outer_px_r:
                if math.hypot(x, y) <= outer_px_r - rad:
                    pts.append((x, y))
                x += pitch
            y += pitch * 0.866
            row += 1
        rng = np.random.default_rng(3)
        keys = np.array([-p[1] + rng.uniform(-0.08, 0.08) * 2 * outer_px_r for p in pts])
        rank = np.empty(len(pts))
        rank[np.argsort(keys)] = np.arange(len(pts))
        rank = rank / max(len(pts) - 1, 1)
        self.a = np.zeros((S, S), np.float32)
        self.c = np.zeros((S, S, 3), np.float32)
        self.order = np.full((S, S), 9.0, np.float32)
        c0 = S / 2.0
        pad = int(math.ceil(rad)) + 2
        for (px, py), rk in zip(pts, rank):
            cx, cy = c0 + px, c0 + py
            ix0, iy0 = int(cx) - pad, int(cy) - pad
            ix1, iy1 = ix0 + 2 * pad + 1, iy0 + 2 * pad + 1
            gx = np.arange(ix0, ix1, dtype=np.float32)[None, :] - np.float32(cx)
            gy = np.arange(iy0, iy1, dtype=np.float32)[:, None] - np.float32(cy)
            dist = np.sqrt(gx * gx + gy * gy)
            al = np.clip(rad - dist + 0.5, 0, 1)
            colp = body_rgb(inner, gx, gy, rad)
            sl = (slice(iy0, iy1), slice(ix0, ix1))
            upd = al > self.a[sl]
            self.a[sl] = np.where(upd, al, self.a[sl])
            self.c[sl] = np.where(upd[..., None], colp, self.c[sl])
            self.order[sl] = np.where(al > 0.02, np.float32(rk), self.order[sl])
        self.count = len(pts)

    def draw(self, frame, frac, alpha=1.0):
        if frac <= 0:
            return
        S = self.S
        m = np.clip((np.float32(frac) - self.order) * 60.0, 0, 1) * self.a * np.float32(alpha)
        x0, y0 = int(540 - S / 2), int(CY - S / 2)
        blend(frame, (x0, x0 + S, y0, y0 + S), self.c, m)


# ---------------------------------------------------------------- the 5 agents
def load_ledger(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_ledger(path, ledger):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)


def ranked_concepts():
    out = []
    for a, A_ in BODIES.items():
        for b, B_ in BODIES.items():
            r = B_["diam"] / A_["diam"]
            if a != b and MIN_RATIO <= r <= MAX_RATIO:
                sweet = 3.0 - min(3.0, abs(math.log10(r) - math.log10(40.0)) * 2.5)
                out.append((A_["fam"] + B_["fam"] + sweet, "SCALE_%s_IN_%s" % (a.upper(), b.upper()), a, b, r))
    out.sort(key=lambda c: (-c[0], c[1]))
    return out


def agent_strategist(ledger, pick=None):
    ranked = ranked_concepts()
    if pick is not None and ranked:
        cands = [ranked[pick % len(ranked)]]
    else:
        cands = [c for c in ranked if c[1] not in ledger]
    if not cands:
        return None
    score, cid, a, b, r = cands[0]
    return dict(concept_id=cid, inner_body=a, outer_body=b, ratio=r, score=round(score, 2),
                target_title="How many %s fit inside %s?" % (BODIES[a]["plural"], nm_mid(BODIES[b]["name"])),
                remaining=len([c for c in ranked if c[1] not in ledger]) - 1)


def _rnd(v):
    if v >= 1e6:
        return "~%.1f million" % (v / 1e6)
    if v >= 1e4:
        return "~{:,}".format(int(round(v, -3)))
    if v >= 1000:
        return "~{:,}".format(int(round(v, -2)))
    return "~%d" % int(round(v))


def agent_researcher(c):
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    ratio = bo["diam"] / bi["diam"]
    vol = ratio ** 3
    if abs(ratio - c["ratio"]) > 1e-6:
        fail("ratio mismatch for " + c["concept_id"], 2)
    if not (MIN_RATIO <= ratio <= MAX_RATIO):
        fail("ratio %.2f outside %.0f-%.0f" % (ratio, MIN_RATIO, MAX_RATIO), 2)
    if bi.get("vol") and bo.get("vol"):
        chk = bo["vol"] / bi["vol"]
        if abs(chk - vol) / vol > 0.02:
            fail("volume cross-check failed: %.4g vs %.4g" % (vol, chk), 2)
    return dict(ratio=ratio, vol_ratio=vol, title=c["target_title"], wide_txt="~%dx" % round(ratio),
                final_txt=_rnd(vol), inner=nm_cap(bi["name"]).replace("The ", ""), outer_cap=nm_cap(bo["name"]))


def agent_director(c, f):
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    A, B = 180.0 / bi["diam"], 940.0 / bo["diam"]
    tint = np.array(bo["col"], np.float32)
    bot = tuple(float(x) for x in np.array((16, 10, 38), np.float32) * 0.85 + tint * 0.15)
    return dict(dur=22.0, hook_end=3.0, zoom_end=9.5, fill_end=18.0, fade_start=20.0, A=A, B=B, gap=1150.0 / A,
                bg_top=(4, 6, 18), bg_bot=bot, title=f["title"],
                beats=[("hook: question + pulsing ring", 0, 3), ("zoom-out reveals container", 3, 9.5),
                       ("fill: copies pack in, counter races", 9.5, 18), ("payoff: final number", 18, 22)])


def agent_engine(c, f, sb, out):
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    inner, outer = c["inner_body"], c["outer_body"]
    ratio, vol = f["ratio"], f["vol_ratio"]
    DUR, z0, z1, z2, zf = sb["dur"], sb["hook_end"], sb["zoom_end"], sb["fill_end"], sb["fade_start"]
    A, B, gap = sb["A"], sb["B"], sb["gap"]
    ri, ro = bi["diam"] / 2, bo["diam"] / 2
    ys_km = ro + gap
    dots = DotLayer(inner, ratio)
    bg = make_bg(sb["bg_top"], sb["bg_bot"], seed=11)
    tf, tl = wrap_fit(sb["title"], 940, 3, 92)
    inner_nm, outer_nm = f["inner"], f["outer_cap"]
    lab_fill = "%s fit inside %s" % (bi["plural"], nm_mid(bo["name"]))

    aud = Audio(DUR)
    aud.add(0.2, whoosh(0.9, 400, 2200, 0.35, 5))
    aud.add(z0 - 0.1, whoosh(2.0, 150, 900, 0.5, 2))
    aud.add(z1, boom())
    for k in range(75):
        aud.add(z1 + (z2 - z1) * sm_inv(k / 75.0), tick(900 + 10 * k, 0.28))
    aud.add(z2 + 0.3, whoosh(1.4, 2200, 300, 0.45, 3))
    aud.add(z2 + 0.35, boom(1.2, 0.6))

    def frame_at(t):
        fr = bg.copy()
        u = sm((t - z0) / (z1 - z0))
        sc = A * (B / A) ** u
        camy = ys_km * u
        sy = CY + (ys_km - camy) * sc
        fe = sm((t - z1) / (z2 - z1))
        in_a = sm(t / 0.5) * (1 - sm((t - (z1 - 2.5)) / 2.0))

        if t >= z0 - 0.2:
            draw_body(fr, outer, 540, sy, ro * sc, 1.0, fe, spin=0.35 * t, t=t)
            dots.draw(fr, fe)

        iy = CY - camy * sc
        draw_body(fr, inner, 540, iy, ri * sc, in_a, spin=0.9 * t, t=t)

        if t < z0 + 0.3:
            pr = 38 + 12 * math.sin(t * 2 * math.pi * 1.5) + ri * sc
            reg = region(540, iy, pr + 4)
            if reg:
                x0, x1, y0, y1 = reg
                dd = np.sqrt((X[:, x0:x1] - 540) ** 2 + (Y[y0:y1, :] - iy) ** 2)
                ring = np.clip(1.8 - np.abs(dd - pr), 0, 1) * np.float32(0.85 * in_a * (1 - sm((t - (z0 - 0.2)) / 0.5)))
                blend(fr, reg, np.array([140, 210, 255], np.float32), ring)

        fr = bloom(fr, 0.40, 160.0, 7)
        sa = 1 - sm((t - zf) / (DUR - zf))
        if sa < 1:
            fr = bg * (1 - sa) + fr * sa

        tx = Text()
        # Top banner backing shadow
        tx.d.rounded_rectangle([60, 100, 1020, 350], 25, fill=(0, 0, 0, int(110 * sm(t / 0.35))))
        tx.lines(W / 2, 225, tl, tf, a=sm(t / 0.35))

        if t < z1 - 1.7:
            tx.put((540, iy + ri * sc + 64), inner_nm, font(46), (160, 215, 255),
                   (1 - sm((t - (z1 - 3.0)) / 1.0)) * sm((t - 0.4) / 0.5))

        # Dynamic Counter System
        if t < z0:
            num, lab = "0", "%s counted so far" % bi["plural"]
        elif t < z1:
            num, lab = f["wide_txt"], "%s is wider than %s" % (outer_nm, inner_nm)
        elif t < z2:
            cur_val = int(vol * fe)
            num, lab = "{:,}".format(cur_val), lab_fill
        else:
            num, lab = f["final_txt"], lab_fill

        ca = 1 - sm((t - (DUR - 1.0)) / 0.9)
        if t >= z2:
            ca *= sm((t - z2) / 0.35)

        # Bottom stat card backing
        tx.d.rounded_rectangle([70, 1500, 1010, 1830], 35, fill=(10, 12, 28, int(190 * ca)))
        tx.d.rounded_rectangle([70, 1500, 1010, 1830], 35, outline=(255, 220, 110, int(140 * ca)), width=3)

        num_col = (255, 230, 110) if t < z2 else (120, 255, 170)
        tx.put((540, 1615), num, fit_font(num, 900, 175, 55), num_col, ca)
        tx.put((540, 1745), lab, fit_font(lab, 900, 52, 24), (240, 245, 255), ca)

        if t >= z1:
            tx.put((540, 1865), "2D slice visualized - Counter reflects 3D volume", font(30), (180, 185, 215),
                   0.95 * sm((t - z1 - 0.5) / 0.6) * ca)

        return tx.apply(np.clip(fr, 0, 255).astype(np.uint8)).tobytes()

    base = os.path.join(out, c["concept_id"].lower())
    render_video(base + ".mp4", DUR, frame_at, aud)
    return base + ".mp4"


def agent_publisher(c, f, sb, mp4, ledger, ledger_path, update_ledger=True):
    quality_check(mp4, sb["dur"])
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    title = c["target_title"] + " #shorts"
    desc = ("%s\nDiameter ratio %.1fx - volume ratio %s.\nDots are a 2D slice; the counter is the 3D volume. "
            "Made with code, no AI video.\n#space #science #scale #shorts" % (c["target_title"], f["ratio"], f["final_txt"]))
    tags = ["space", "science", "size comparison", "scale", bi["name"].lower(), bo["name"].lower(), "shorts", "how big"]
    write_meta(os.path.splitext(mp4)[0] + ".txt", title, desc, tags,
               "Check diameters in bodies.json against a primary source before relying on them.")
    if update_ledger and c["concept_id"] not in ledger:
        ledger.append(c["concept_id"])
        save_ledger(ledger_path, ledger)
    print("\n=== YOUTUBE METADATA ===\nTITLE: %s\nDESCRIPTION:\n%s\nTAGS: %s\n========================" % (title, desc, ", ".join(tags)))
    print("ledger: %d concepts used, %d left" % (len(ledger), c["remaining"]))


def run_pipeline(out, ledger_path, pick=None):
    ledger = load_ledger(ledger_path)
    c = agent_strategist(ledger, pick)
    if c is None:
        print("NO_NEW_CONCEPT: every valid pair is already in the ledger. Add bodies to data/bodies.json.")
        return None
    print("1 strategist:", c["concept_id"], "| score", c["score"], "| ratio %.2f" % c["ratio"])
    f = agent_researcher(c)
    print("2 researcher: wide %s, volume %s" % (f["wide_txt"], f["final_txt"]))
    sb = agent_director(c, f)
    print("3 director:", " | ".join("%s %.1f-%.1fs" % (n.split(":")[0], a_, b_) for n, a_, b_ in sb["beats"]))
    mp4 = agent_engine(c, f, sb, out)
    agent_publisher(c, f, sb, mp4, ledger, ledger_path, update_ledger=pick is None)
    return mp4


# =============================================================== ENGINE 2: ILLUSION
def _panel(S=2):
    img = Image.new("RGB", (W * S, H * S), (12, 14, 34))
    return img, ImageDraw.Draw(img)


def _grad_bg(S=2):
    t = np.linspace(0, 1, H * S, dtype=np.float32)[:, None, None]
    arr = (np.array((10, 12, 32), np.float32) * (1 - t) + np.array((26, 16, 56), np.float32) * t) * np.ones((1, W * S, 1), np.float32)
    return Image.fromarray(arr.astype(np.uint8))


PX0, PX1, PY0, PY1 = 70, 1010, 560, 1420


def ill_muller(rng, S=2):
    img = _grad_bg(S)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([PX0 * S, PY0 * S, PX1 * S, PY1 * S], 40 * S, fill=(246, 246, 250))
    L = int(rng.integers(470, 540))
    ang = math.radians(rng.uniform(28, 40))
    fl = int(rng.integers(70, 95))
    x0, x1 = 540 - L // 2, 540 + L // 2
    ys = (860, 1140)
    wd = 9 * S
    col = (30, 30, 50)
    for k, y in enumerate(ys):
        d.line([x0 * S, y * S, x1 * S, y * S], fill=col, width=wd)
        sgn = -1 if k == 0 else 1
        for xe, dirx in ((x0, -1), (x1, 1)):
            for sy in (-1, 1):
                d.line([xe * S, y * S, (xe + sgn * dirx * fl * math.cos(ang)) * S, (y + sy * fl * math.sin(ang)) * S],
                       fill=col, width=wd)
    f = font(64 * S)
    for lab, y in (("A", ys[0]), ("B", ys[1])):
        d.text((130 * S, y * S), lab, font=f, fill=(200, 60, 80), anchor="mm")
    meta = dict(x0=x0, x1=x1, ys=ys)

    def reveal(dr, p, S=1):
        for xx in (meta["x0"], meta["x1"]):
            yb = 760 + (1240 - 760) * p
            for y in range(760, int(yb), 28):
                dr.line([xx, y, xx, min(y + 16, yb)], fill=(255, 70, 90), width=6)
    return dict(img=img, reveal=reveal, q="Which line is longer: A or B?", ans="They are EQUAL!",
                exp=["The fins make your brain judge", "length by context, not by size."], opts="Comment A or B")


def ill_ebbinghaus(rng, S=2):
    img = _grad_bg(S)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([PX0 * S, PY0 * S, PX1 * S, PY1 * S], 40 * S, fill=(246, 246, 250))
    swap = bool(rng.integers(0, 2))
    rc = 64
    cy = 990
    cl, cr = (340, 780) if not swap else (300, 740)
    big_c, small_c = (70, 110, 200), (70, 110, 200)

    def ring(cx, n, rr, dist):
        for k in range(n):
            a = 2 * math.pi * k / n
            x, y = cx + dist * math.cos(a), cy + dist * math.sin(a)
            d.ellipse([(x - rr) * S, (y - rr) * S, (x + rr) * S, (y + rr) * S], fill=big_c)
    big_x, small_x = (cl, cr) if not swap else (cr, cl)
    ring(big_x, 6, 88, rc + 88 + 18)
    ring(small_x, 8, 22, rc + 22 + 16)
    for cx in (cl, cr):
        d.ellipse([(cx - rc) * S, (cy - rc) * S, (cx + rc) * S, (cy + rc) * S], fill=(255, 140, 0))
    meta = dict(cl=cl, cr=cr, cy=cy, rc=rc)

    def reveal(dr, p, S=1):
        pr = sm(p)
        for cx in (meta["cl"], meta["cr"]):
            rr = meta["rc"] + 2
            dr.ellipse([cx - rr, meta["cy"] - rr, cx + rr, meta["cy"] + rr], outline=(20, 20, 40), width=6)
        x_a, x_b = meta["cl"] - meta["rc"], meta["cr"] + meta["rc"]
        xm = x_a + (x_b - x_a) * pr
        dr.line([x_a, meta["cy"] + 150, xm, meta["cy"] + 150], fill=(255, 70, 90), width=7)
        for xx in (meta["cl"] - meta["rc"], meta["cl"] + meta["rc"], meta["cr"] - meta["rc"], meta["cr"] + meta["rc"]):
            dr.line([xx, meta["cy"] + 130, xx, meta["cy"] + 170], fill=(255, 70, 90), width=5)
    return dict(img=img, reveal=reveal, q="Which orange circle is bigger?", ans="Exactly the same size!",
                exp=["Your brain compares each circle", "with the ones around it."], opts="Comment LEFT or RIGHT")


def ill_contrast(rng, S=2):
    img = _grad_bg(S)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([PX0 * S, PY0 * S, PX1 * S, PY1 * S], 40 * S, fill=(246, 246, 250))
    gray = int(rng.integers(112, 140))
    dark, light = int(rng.integers(20, 45)), int(rng.integers(215, 240))
    mid = 540
    swap = bool(rng.integers(0, 2))
    lcol, rcol = ((dark,) * 3, (light,) * 3) if not swap else ((light,) * 3, (dark,) * 3)
    d.rounded_rectangle([PX0 * S, PY0 * S, mid * S, PY1 * S], 40 * S, fill=lcol)
    d.rectangle([(PX0 + 60) * S, PY0 * S, mid * S, PY1 * S], fill=lcol)
    d.rounded_rectangle([mid * S, PY0 * S, PX1 * S, PY1 * S], 40 * S, fill=rcol)
    d.rectangle([mid * S, PY0 * S, (PX1 - 60) * S, PY1 * S], fill=rcol)
    g = (gray,) * 3
    bars = ((270, 470), (610, 810))
    for x0, x1 in bars:
        d.rectangle([x0 * S, 880 * S, x1 * S, 1110 * S], fill=g)
    meta = dict(g=g, bars=bars)

    def reveal(dr, p, S=1):
        half = (610 - 470) / 2 * sm(p)
        dr.rectangle([540 - half - 1, 880, 540 + half + 1, 1110], fill=meta["g"])
    return dict(img=img, reveal=reveal, q="Which gray bar is darker?", ans="Exactly the same gray!",
                exp=["Your eyes judge brightness", "against the background."], opts="Comment LEFT or RIGHT")


def ill_cafe(rng, S=2):
    img = _grad_bg(S)
    d = ImageDraw.Draw(img)
    pw, ph = PX1 - PX0, PY1 - PY0
    tile = Image.new("RGB", (pw * S, ph * S), (128, 128, 128))
    td = ImageDraw.Draw(tile)
    w = 94
    rows_h = 56
    mort = 6
    nrows = ph // rows_h
    shift = w / 2.0 if rng.random() < 0.5 else w / 2.0
    for r in range(nrows + 1):
        y0 = r * rows_h
        off = -(shift if r % 2 else 0)
        for c in range(-1, pw // w + 3):
            x0 = off + c * w
            colr = (15, 15, 20) if (c % 2 == 0) else (248, 248, 250)
            td.rectangle([x0 * S, (y0 + mort / 2) * S, (x0 + w) * S - 1, (y0 + rows_h - mort / 2) * S - 1], fill=colr)
    mask = Image.new("L", (pw * S, ph * S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, pw * S, ph * S], 40 * S, fill=255)
    img.paste(tile, (PX0 * S, PY0 * S), mask)
    meta = dict(rows=[PY0 + r * rows_h for r in range(1, nrows)])

    def reveal(dr, p, S=1):
        xe = PX0 + (PX1 - PX0) * sm(p)
        for i, y in enumerate(meta["rows"][::3][:5]):
            dr.line([PX0, y, xe, y], fill=(255, 70, 90) if i % 2 == 0 else (70, 200, 255), width=6)
    return dict(img=img, reveal=reveal, q="Are the gray lines sloped or straight?", ans="All perfectly parallel!",
                exp=["The shifted tiles fool your", "brain into seeing slopes."], opts="Comment STRAIGHT or SLOPED")


ILLUSIONS = [("muller_lyer", ill_muller), ("ebbinghaus", ill_ebbinghaus), ("contrast", ill_contrast), ("cafe_wall", ill_cafe)]


def engine_illusion(seed, out):
    name, builder = ILLUSIONS[seed % len(ILLUSIONS)]
    rng = np.random.default_rng(1000 + seed)
    ill = builder(rng)
    base_img = ill["img"].resize((W, H), Image.LANCZOS)
    DUR = 22.0
    print("illusion:", name, "|", ill["q"])
    tf, tl = wrap_fit(ill["q"], 940, 3, 88)
    ef = fit_font(max(ill["exp"], key=len), 940, 56, 28)

    aud = Audio(DUR)
    aud.add(0.1, whoosh(0.8, 500, 2500, 0.35, 4))
    for k, tt in enumerate((6.5, 7.5, 8.5)):
        aud.add(tt, tick(900 + 200 * k, 0.4))
    aud.add(9.2, bell(784, 1.4, 0.45))
    aud.add(9.2, bell(988, 1.4, 0.3))
    aud.add(14.2, tick(1200, 0.25))

    def frame_at(t):
        img = base_img.copy()
        dr = ImageDraw.Draw(img)
        if t >= 9.2:
            ill["reveal"](dr, sm((t - 9.2) / 3.0), 1)
        arr = np.asarray(img, np.float32)
        fade = sm(t / 0.5) * (1 - sm((t - 20.5) / 1.4))
        bg = make_bg_cached()
        arr = bg * (1 - fade) + arr * fade
        tx = Text()
        tx.lines(W / 2, 235, tl, tf, a=sm(t / 0.35) * (1 - sm((t - 21.0) / 0.9)))
        if 2.5 <= t < 9.2:
            tx.put((540, 1560), ill["opts"], fit_font(ill["opts"], 940, 60), (255, 226, 120), sm((t - 2.5) / 0.4))
            k = min(max((t - 2.5) / 6.7, 0), 1)
            tx.d.rounded_rectangle([140, 1650, 940, 1676], 13, fill=(255, 255, 255, 60))
            tx.d.rounded_rectangle([140, 1650, 140 + 800 * (1 - k), 1676], 13, fill=(255, 226, 120, 230))
            if t >= 6.2:
                n = 3 - int((t - 6.2) / 1.0)
                if n >= 1:
                    tx.put((540, 1480), str(n), font(110), (255, 255, 255), 1.0)
        elif t < 2.5:
            tx.put((540, 1560), "Look closely...", font(64), (235, 240, 255), sm(t / 0.5))
        if 9.2 <= t:
            a = sm((t - 9.5) / 0.5) * (1 - sm((t - 21.0) / 0.9))
            tx.put((540, 1530), ill["ans"], fit_font(ill["ans"], 940, 92, 40), (120, 255, 160), a)
        if 14.2 <= t:
            a = sm((t - 14.2) / 0.6) * (1 - sm((t - 21.0) / 0.9))
            tx.lines(540, 1700, ill["exp"], ef, (235, 240, 255), a, lh=int(ef.size * 1.2))
        return tx.apply(np.clip(arr, 0, 255).astype(np.uint8)).tobytes()

    _bg = [None]

    def make_bg_cached():
        if _bg[0] is None:
            _bg[0] = np.asarray(_grad_bg(1), np.float32)
        return _bg[0]

    base = os.path.join(out, "illusion_%s_%d" % (name, seed))
    render_video(base + ".mp4", DUR, frame_at, aud)
    quality_check(base + ".mp4", DUR)
    write_meta(base + ".txt", ill["q"] + " #shorts #illusion",
               "%s\nAnswer: %s %s\nMade with code. Comment your guess!\n#opticalillusion #brain #shorts"
               % (ill["q"], ill["ans"], " ".join(ill["exp"])),
               ["optical illusion", "brain test", "perception", "illusion", "shorts", "can you see it"])
    return base + ".mp4"


# =============================================================== ENGINE 3: PHYSICS
PAL = [("Red", (255, 70, 80)), ("Orange", (255, 160, 40)), ("Yellow", (250, 230, 70)), ("Green", (80, 230, 120)),
       ("Cyan", (70, 220, 255)), ("Blue", (90, 130, 255)), ("Purple", (185, 105, 255)), ("Pink", (255, 115, 205))]
ARENA_R, BALL_R = 430.0, 32.0


def simulate(seed, n, theta, omega, speed=540.0, T=25.0):
    R, r = ARENA_R, BALL_R
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 2))
    for i in range(n):
        for _ in range(800):
            a, dd = rng.uniform(0, 2 * math.pi), rng.uniform(0, R - 3 * r)
            p = np.array([dd * math.cos(a), dd * math.sin(a)])
            if all(np.hypot(*(p - pos[j])) > 2.4 * r for j in range(i)):
                pos[i] = p
                break
    ang = rng.uniform(0, 2 * math.pi, n)
    vel = np.stack([np.cos(ang), np.sin(ang)], 1) * speed
    phi0 = rng.uniform(0, 2 * math.pi)
    sub = 8
    dt = 1.0 / (FPS * sub)
    nf = int(T * FPS)
    status = np.zeros(n, int)
    tel = np.full(n, np.nan)
    states = np.zeros((nf, n, 2))
    stat = np.zeros((nf, n), int)
    events = []
    for f in range(nf):
        for s in range(sub):
            t = (f * sub + s) * dt
            alive = int((status == 0).sum())
            th = theta if alive > 1 else 0.0
            phi = phi0 + omega * t
            pos += vel * dt
            for i in range(n):
                if status[i]:
                    continue
                x, y = pos[i]
                d = math.hypot(x, y)
                if d > R - r:
                    a = math.atan2(y, x)
                    diff = (a - phi + math.pi) % (2 * math.pi) - math.pi
                    if abs(diff) < th / 2 and alive > 1:
                        status[i], tel[i] = 1, t
                        alive -= 1
                        vel[i] = np.array([x, y]) / d * speed * 1.1
                        events.append((t, "out", i, -1, speed))
                        continue
                    nx, ny = x / d, y / d
                    vn = vel[i, 0] * nx + vel[i, 1] * ny
                    if vn > 0:
                        vel[i] -= 2 * vn * np.array([nx, ny])
                        events.append((t, "wall", i, -1, vn))
                    pos[i] = [nx * (R - r), ny * (R - r)]
            for i in range(n):
                if status[i]:
                    continue
                for j in range(i + 1, n):
                    if status[j]:
                        continue
                    dl = pos[j] - pos[i]
                    dist = math.hypot(dl[0], dl[1])
                    if 1e-6 < dist < 2 * r:
                        nrm = dl / dist
                        push = (2 * r - dist) / 2
                        pos[i] -= nrm * push
                        pos[j] += nrm * push
                        vn = float(np.dot(vel[i] - vel[j], nrm))
                        if vn > 0:
                            vel[i] -= vn * nrm
                            vel[j] -= vn * nrm
                            events.append((t, "ball", i, j, vn))
            for i in range(n):
                if not status[i]:
                    sp = math.hypot(vel[i, 0], vel[i, 1])
                    if sp > 1e-6:
                        vel[i] *= speed / sp
        states[f] = pos
        stat[f] = status
    return dict(states=states, stat=stat, tel=tel, events=events, theta=theta, phi0=phi0, omega=omega, n=n)


def pretest(seed, n=8):
    best, best_score = None, 1e9
    thetas = [0.38, 0.5, 0.62, 0.75, 0.9, 0.3]
    for k in range(90):
        th = thetas[k % len(thetas)]
        sim = simulate(seed * 1000 + k, n, th, 0.9 + 0.1 * (k % 5))
        te = np.sort(sim["tel"][~np.isnan(sim["tel"])])
        if len(te) < n - 1:
            continue
        last, prev, first = te[-1], te[-2], te[0]
        coll = sum(1 for e in sim["events"] if e[1] == "ball")
        score = abs(last - 18.0) + (0 if prev <= last - 1.0 else 5) + (0 if first >= 2.0 else 5) + (0 if coll >= 20 else 5)
        ok = 15.0 <= last <= 21.5 and last - prev >= 1.0 and first >= 2.0 and coll >= 20
        print("  pretest %02d theta=%.2f last=%.1fs margin=%.1fs first=%.1fs collisions=%d -> %s"
              % (k, th, last, last - prev, first, coll, "PASS" if ok else "reject"))
        if ok:
            return sim
        if score < best_score:
            best, best_score = sim, score
    fail("no interesting simulation found for this seed (try another --seed)", 7)


def engine_physics(seed, out):
    n = 8
    sim = pretest(seed, n)
    DUR = 25.0
    states, stat, tel = sim["states"], sim["stat"], sim["tel"]
    win = int(np.where(stat[-1] == 0)[0][0])
    te = float(np.nanmax(tel))
    wname, wcol = PAL[win]
    print("physics: winner", wname, "| last elimination %.1fs" % te)

    S = 1000
    cx0, cy0 = 540.0, float(CY)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    ddx, ddy = xx - S / 2, yy - S / 2
    dist = np.sqrt(ddx * ddx + ddy * ddy)
    ang = np.arctan2(ddy, ddx)
    ring_a = np.clip(7.0 - np.abs(dist - ARENA_R), 0, 1)
    ring_glow = np.exp(-np.abs(dist - ARENA_R) / 14.0) * 0.5
    bg = make_bg((5, 6, 20), (16, 10, 40), seed=5, stars=160)
    for rr in (130, 260):
        reg = region(cx0, cy0, rr + 3)
        x0, x1, y0, y1 = reg
        dd = np.sqrt((X[:, x0:x1] - cx0) ** 2 + (Y[y0:y1, :] - cy0) ** 2)
        blend(bg, reg, np.array([60, 70, 130], np.float32), np.clip(1.4 - np.abs(dd - rr), 0, 1) * 0.25)
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n):
        if not np.isnan(tel[i]):
            fi = min(int(tel[i] * FPS), len(states) - 1)
            p0 = states[fi, i]
            for _ in range(26):
                a, sp = rng.uniform(0, 2 * math.pi), rng.uniform(150, 520)
                parts.append((tel[i], p0[0], p0[1], sp * math.cos(a), sp * math.sin(a), i))
    trail = np.zeros((S, S, 3), np.float32)

    aud = Audio(DUR)
    aud.add(0.1, whoosh(0.8, 400, 2400, 0.35, 8))
    last = {}
    for (t, kind, i, j, v) in sim["events"]:
        if kind == "out":
            aud.add(t, pop(0.55))
            aud.add(t, whoosh(0.7, 2200, 300, 0.35, 20 + i))
        elif kind == "wall":
            if t - last.get(("w", i), -1) > 0.09:
                aud.add(t, bell(note(i), 0.45, 0.16))
                last[("w", i)] = t
        else:
            if t - last.get(("b", i, j), -1) > 0.09:
                aud.add(t, bell(note(i), 0.4, 0.2))
                aud.add(t, bell(note(j) * 1.0, 0.4, 0.2))
                last[("b", i, j)] = t
    for k, kk in enumerate((0, 2, 4, 7)):
        aud.add(te + 0.35 + 0.16 * k, bell(note(kk, 392.0), 1.0, 0.4))

    tf = fit_font("Which color survives?", 940, 96)

    def frame_at(t):
        f = bg.copy()
        fi = min(int(t * FPS), len(states) - 1)
        pos, st = states[fi], stat[fi]
        alive_n = int((st == 0).sum())
        th = sim["theta"] if alive_n > 1 else 0.0
        phi = sim["phi0"] + sim["omega"] * t
        diff = (ang - phi + math.pi) % (2 * math.pi) - math.pi
        gmask = np.clip((np.abs(diff) - th / 2) * (ARENA_R / 6.0), 0, 1)
        edge = np.clip(1 - np.abs(np.abs(diff) - th / 2) * (ARENA_R / 14.0), 0, 1) * (th > 0)
        ra = ring_a * gmask + ring_glow * gmask * 0.6
        rcol = np.array([70, 200, 255], np.float32) * (0.85) + 120 * edge[..., None] * np.array([1, 0.6, 0.3], np.float32)
        reg = (int(cx0 - S / 2), int(cx0 + S / 2), int(cy0 - S / 2), int(cy0 + S / 2))
        blend(f, reg, rcol, np.clip(ra + edge * 0.9 * ring_a, 0, 1))
        trail[...] *= 0.86
        for i in range(n):
            if st[i] == 0:
                bx, by = pos[i][0] + S / 2, pos[i][1] + S / 2
                x0, x1, y0, y1 = int(bx - 44), int(bx + 44), int(by - 44), int(by + 44)
                if x0 >= 0 and y0 >= 0 and x1 < S and y1 < S:
                    dd = np.sqrt((xx[y0:y1, x0:x1] - bx) ** 2 + (yy[y0:y1, x0:x1] - by) ** 2)
                    m = np.clip(1 - dd / (BALL_R * 1.25), 0, 1) ** 1.5 * 0.5
                    trail[y0:y1, x0:x1] += m[..., None] * np.array(PAL[i][1], np.float32) * 0.55
        sub = f[int(cy0 - S / 2):int(cy0 + S / 2), int(cx0 - S / 2):int(cx0 + S / 2)]
        sub += np.clip(trail, 0, 255)
        for i in range(n):
            px, py = cx0 + pos[i][0], cy0 + pos[i][1]
            if st[i] == 0:
                disc(f, px, py, BALL_R, PAL[i][1])
            else:
                age = t - tel[i]
                al = 1 - sm(age / 1.1)
                disc(f, px, py, BALL_R * (1 - 0.3 * sm(age / 1.1)), PAL[i][1], al)
        for (t0, x0_, y0_, vx, vy, i) in parts:
            age = t - t0
            if 0 <= age < 0.8:
                disc(f, cx0 + x0_ + vx * age, cy0 + y0_ + vy * age, 7 * (1 - age / 0.8), PAL[i][1], 1 - age / 0.8, False)
        f = bloom(f, 0.6, 150.0, 7)
        tx = Text()
        tx.put((540, 190), "Which color survives?", tf, a=sm(t / 0.4))
        if t < 2.5:
            tx.put((540, 300), "Pick one in the comments!", font(52), (255, 226, 120), sm(t / 0.5))
        arr = np.clip(f, 0, 255)
        for i in range(n):
            lx = 140 + i * 114.3
            alive_i = st[i] == 0
            disc(arr, lx, 1560, 30, PAL[i][1], 1.0 if alive_i else 0.18, False)
        if t >= te + 0.3:
            a = sm((t - te - 0.3) / 0.5)
            tx.put((540, 1700), "%s survives!" % wname, fit_font("%s survives!" % wname, 940, 120), wcol, a)
        else:
            tx.put((540, 1700), "Alive: %d" % alive_n, font(110), (255, 226, 120), 1.0)
        return tx.apply(arr.astype(np.uint8)).tobytes()

    base = os.path.join(out, "physics_survival_%d" % seed)
    render_video(base + ".mp4", DUR, frame_at, aud)
    quality_check(base + ".mp4", DUR)
    write_meta(base + ".txt", "Which color survives? #shorts",
               "Eight balls, one escape gap. Which color survives? Comment your pick!\nMade with code (own physics + original sounds).\n#satisfying #physics #shorts",
               ["satisfying", "ball simulation", "survival", "physics", "shorts", "which color"],
               "Run %d: winner %s (simulation pre-tested: elimination timing and collisions in range)." % (seed, wname))
    return base + ".mp4"


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["auto", "scale", "illusion", "physics", "all"])
    ap.add_argument("--seed", type=int, default=-1, help="specific variant (scale: index into ranked list, no ledger write)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--bodies", default="data/bodies.json")
    ap.add_argument("--ledger", default="ledger.json")
    ap.add_argument("--count", type=int, default=1, help="auto/scale: how many videos")
    a = ap.parse_args()
    for t in ("ffmpeg", "ffprobe"):
        if shutil.which(t) is None:
            fail("%s not found on PATH" % t, 3)
    os.makedirs(a.out, exist_ok=True)
    load_bodies(a.bodies)
    print("bodies loaded: %d (%s)" % (len(BODIES), a.bodies if os.path.exists(a.bodies) else "built-in defaults"))
    other = {"illusion": engine_illusion, "physics": engine_physics}
    todo = {"auto": ["scale"], "scale": ["scale"], "all": ["scale", "illusion", "physics"]}.get(a.engine, [a.engine])
    for e in todo:
        for k in range(a.count):
            t0 = time.time()
            print("== %s ==" % e)
            if e == "scale":
                path = run_pipeline(a.out, a.ledger, None if a.seed < 0 else a.seed + k)
            else:
                seed = a.seed if a.seed >= 0 else len([x for x in os.listdir(a.out) if x.startswith(e) and x.endswith(".mp4")])
                path = other[e](seed, a.out)
            if path:
                print("done -> %s  (%.0fs)\n" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
