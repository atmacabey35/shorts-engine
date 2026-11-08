#!/usr/bin/env python3
"""
Shorts engine - True 3D Volumetric Depth & Parallax rendering with NumPy + Pillow + FFmpeg.
No AI video, no paid API, zero crash.
"""
import sys, os, math, json, shutil, subprocess, tempfile, wave, argparse, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, SR = 1080, 1920, 30, 44100
CY = 980
OUTDIR = "out"


def fail(msg, code=1):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(code)


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    return a + (b - a) * u


# =============================================================== text & typography
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
            for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (3, 5), (-3, 3)):
                self._draw((xy[0] + dx, xy[1] + dy), text, f, (0, 0, 0, int(al * 0.90)))
        self._draw(xy, text, f, (col[0], col[1], col[2], al))

    def lines(self, cx, cy, lines, f, col=(255, 255, 255), a=1.0, lh=None):
        lh = lh or int(getattr(f, "size", 60) * 1.18)
        y0 = cy - (len(lines) - 1) * lh / 2
        for i, l in enumerate(lines):
            self.put((cx, y0 + i * lh), l, f, col, a)

    def apply(self, rgb_u8):
        base = Image.fromarray(rgb_u8).convert("RGBA")
        return Image.alpha_composite(base, self.img).convert("RGB")


# =============================================================== background / glow / stars
def make_bg(top, bot, seed=7, stars=450, vignette=0.68):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    bg = (np.array(top, np.float32) * (1 - t) + np.array(bot, np.float32) * t) * np.ones((1, W, 1), np.float32)
    for _ in range(stars):
        sx, sy = int(rng.integers(0, W - 2)), int(rng.integers(0, H - 2))
        v = float(rng.integers(110, 255))
        s = 2 if rng.random() < 0.28 else 1
        bg[sy:sy + s, sx:sx + s, :] = np.maximum(bg[sy:sy + s, sx:sx + s, :], v)
    if vignette:
        yy = (np.arange(H, dtype=np.float32)[:, None] - H / 2) / (H / 2)
        xx = (np.arange(W, dtype=np.float32)[None, :] - W / 2) / (W / 2)
        v = 1 - vignette * np.clip((xx * xx * 0.70 + yy * yy * 0.60) - 0.16, 0, 1)
        bg *= v[..., None]
    return bg


def bloom(frame, strength=0.55, thresh=150.0, radius=8):
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


def ambient_sub_bass(dur, f0=55.0, amp=0.32):
    n = int(dur * SR)
    t = np.arange(n) / SR
    mod = 1.0 + 0.15 * np.sin(2 * np.pi * 0.25 * t)
    drone = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * (f0 * 1.5) * t)
    return drone * mod * amp


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


_grid = np.linspace(0, 1, 2001)
_sm = _grid * _grid * (3 - 2 * _grid)


def sm_inv(y):
    return float(np.interp(y, _sm, _grid))


# =============================================================== writer + QC
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
    if bad:
        fail("quality check failed: " + "; ".join(bad), 5)
    print("QC OK: %dx%d h264/yuv420p %dfps, audio, %.2fs, frames not blank" % (W, H, FPS, got))


def write_meta(path_txt, title, desc, tags, notes=""):
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("TITLE:\n%s\n\nDESCRIPTION:\n%s\n\nTAGS:\n%s\n" % (title, desc, ", ".join(tags)))
        if notes:
            f.write("\nNOTES:\n%s\n" % notes)


# =============================================================== ENGINE 1: SCALE (3D Volumetric)
DEFAULT_RAW = {
    "moon": {"name": "The Moon", "plural": "Moons", "diam_km": 3474.8, "color_hex": "#B0B0AC"},
    "mercury": {"name": "Mercury", "plural": "Mercuries", "diam_km": 4879.4, "color_hex": "#A08E7E"},
    "mars": {"name": "Mars", "plural": "Mars-sized worlds", "diam_km": 6779.0, "color_hex": "#C4623E"},
    "earth": {"name": "Earth", "plural": "Earths", "diam_km": 12742.0, "color_hex": "#2364D7"},
    "neptune": {"name": "Neptune", "plural": "Neptunes", "diam_km": 49244.0, "color_hex": "#3C64D7"},
    "saturn": {"name": "Saturn", "plural": "Saturns", "diam_km": 116460.0, "color_hex": "#CEB078"},
    "jupiter": {"name": "Jupiter", "plural": "Jupiters", "diam_km": 139820.0, "color_hex": "#B0764E"},
    "sun": {"name": "The Sun", "plural": "Suns", "diam_km": 1392700.0, "color_hex": "#FF9D00"},
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
                fam=float(raw.get("familiarity", 3)), core=(255, 240, 180) if style == "sun" else _lighten(col, 0.65),
                edge=(255, 90, 15) if style == "sun" else (int(col[0] * 0.98), int(col[1] * 0.52), int(col[2] * 0.15)))


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


# ---- Procedural Shaders
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
    lit = 0.10 + 0.90 * (diff ** 0.80)
    st = b["style"]
    base = np.array(b["col"], np.float32)

    if st == "earth":
        h = fbm(px, py, pz, 1.5, 5)
        land = sstep(0.12, 0.2, h)
        shallow = sstep(-0.1, 0.14, h)
        ocean = np.array([4, 30, 105], np.float32) * (1 - shallow[..., None]) + np.array([20, 110, 190], np.float32) * shallow[..., None]
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
        return out + np.array([90, 160, 255], np.float32) * rim[..., None] * 1.25

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
        return out + np.array(b["col2"], np.float32) * rim[..., None] * 0.70

    if st == "sun":
        gran = fbm(px, py, pz, 5.0, 4, off=t * 0.4)
        flare_noise = fbm(px, py, pz, 10.0, 3, off=t * 0.8) * 0.18
        limb = np.clip((1 - z) ** 1.1, 0, 1)
        core_col = np.array([255, 240, 175], np.float32)
        mid_col = np.array([255, 140, 20], np.float32)
        edge_col = np.array([245, 60, 0], np.float32)
        col = core_col * (1 - limb[..., None]) + mid_col * limb[..., None]
        col = col * (0.92 + 0.16 * (gran + flare_noise))[..., None]
        col += edge_col * ((1 - z) ** 2.5)[..., None] * 0.85
        return col

    # Rock
    h = fbm(px, py, pz, 2.2, 5)
    alb = 0.70 + 0.30 * h
    cr = np.sin(px * 7.5) * np.sin(py * 7.5 + 1.0) * np.sin(pz * 7.5 - 0.5)
    crater = sstep(0.35, 0.7, cr)
    rimc = sstep(0.2, 0.35, cr) * (1 - sstep(0.35, 0.5, cr))
    col = base * (alb * (1 - 0.35 * crater) + 0.22 * rimc)[..., None]
    rim = (1 - z) ** 3.0 * (0.2 + 0.8 * diff)
    return col * lit[..., None] + np.array(b["col"], np.float32) * rim[..., None] * 0.30


def draw_body(frame, key, cx, cy, r, alpha=1.0, darken=0.0, spin=0.0, t=0.0, glass_mode=False):
    """Draws celestial body or translucent atmospheric glass shell."""
    if alpha <= 0.01:
        return
    sun = BODIES[key]["style"] == "sun"
    ext = r * 2.2 if sun else r * 1.35
    r = max(r, 2.4)
    reg = region(cx, cy, ext)
    if reg is None:
        return
    x0, x1, y0, y1 = reg
    dx = X[:, x0:x1] - np.float32(cx)
    dy = Y[y0:y1, :] - np.float32(cy)
    d = np.sqrt(dx * dx + dy * dy)
    inside = np.clip(r - d + 0.5, 0, 1)

    # 1. Atmospheric Glow / Corona
    glow_col = np.array([255, 120, 25] if sun else BODIES[key]["col2"], np.float32)
    glow_dist = d - r
    glow_a = np.where(glow_dist > 0, np.exp(-glow_dist / (0.22 * r)), 0.0).astype(np.float32)
    blend(frame, reg, glow_col, glow_a * (0.90 if sun else 0.65) * alpha)

    # 2. Body Shading or Glass Shell
    if glass_mode:
        # Realistic Glass / Translucent atmospheric container:
        # Highly transparent center, luminous Fresnel rim ring showing the true boundary!
        z_norm = np.clip(d / r, 0, 1)
        fresnel = (z_norm ** 4.5) * 0.95 + 0.08
        shell_col = np.array(BODIES[key]["col2"] if not sun else [255, 160, 40], np.float32)
        blend(frame, reg, shell_col, inside * fresnel * alpha)
        # Outer delicate neon boundary
        edge_line = np.clip(1.5 - np.abs(d - r), 0, 1) * 0.85
        blend(frame, reg, np.array([255, 255, 255], np.float32), edge_line * alpha)
    else:
        h_, w_ = d.shape
        half = h_ * w_ > 600000
        def shade(dxs, dys):
            return body_rgb(key, dxs, dys, r, spin, t)
        if half:
            col = shade(dx[:, ::2], dy[::2, :])
            col = np.repeat(np.repeat(col, 2, axis=0), 2, axis=1)[:h_, :w_]
        else:
            col = shade(dx, dy)
        blend(frame, reg, col, inside * alpha)


# ---- 3D Volumetric Sphere Packing Layer (Z-Depth + Real Orbital Parallax)
class VolumetricSphere3D:
    def __init__(self, inner_key, ratio, outer_px_r=460.0):
        self.inner_key = inner_key
        # Calculate visual packing capacity and 3D spherical positions
        max_balls = min(320, int(ratio * 2.8))
        rng = np.random.default_rng(42)
        
        # Fibonacci / Poisson spherical volume sampling (x, y, z inside unit sphere)
        pts = []
        for _ in range(max_balls * 6):
            if len(pts) >= max_balls:
                break
            # Random point inside unit sphere
            u = rng.uniform(0, 1) ** (1/3) * 0.92
            costheta = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * math.pi)
            sintheta = math.sqrt(max(0.0, 1.0 - costheta * costheta))
            p = np.array([u * sintheta * math.cos(phi), u * sintheta * math.sin(phi), u * costheta])
            if all(np.linalg.norm(p - ep) > 0.16 for ep in pts):
                pts.append(p)
                
        self.pts = np.array(pts, np.float32)
        # Order by distance from bottom-back to top-front (filling effect)
        fill_order = self.pts[:, 1] + 0.3 * self.pts[:, 2]
        self.ranks = np.argsort(fill_order)
        self.base_r = max(11.0, min(36.0, outer_px_r / (ratio ** 0.45)))

    def draw(self, frame, frac, alpha=1.0, center_x=540, center_y=CY, outer_px_r=460.0, cam_orbit=0.0):
        if frac <= 0.0 or alpha <= 0.01:
            return
        n_visible = int(len(self.pts) * frac)
        if n_visible == 0:
            return

        visible_indices = self.ranks[:n_visible]
        sub_pts = self.pts[visible_indices]

        # 3D Orbit Rotation around Y axis
        cos_o, sin_o = math.cos(cam_orbit), math.sin(cam_orbit)
        rot_x = sub_pts[:, 0] * cos_o - sub_pts[:, 2] * sin_o
        rot_z = sub_pts[:, 0] * sin_o + sub_pts[:, 2] * cos_o
        rot_y = sub_pts[:, 1]

        # Sort strictly by Z (back to front) for true 3D depth compositing!
        z_sort = np.argsort(rot_z)
        
        for idx in z_sort:
            x_norm = rot_x[idx]
            y_norm = rot_y[idx]
            z_norm = rot_z[idx] # from -1.0 (back) to +1.0 (front)

            # Perspective projection: back spheres are smaller and dimmer
            depth_scale = 1.0 + 0.35 * z_norm
            depth_dim = 0.55 + 0.45 * (z_norm + 1.0) / 2.0

            px = center_x + x_norm * outer_px_r
            py = center_y + y_norm * outer_px_r
            pr = self.base_r * depth_scale

            disc(frame, px, py, pr, BODIES[self.inner_key]["col"], alpha=alpha * depth_dim, glossy=True)


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
    return dict(ratio=ratio, vol_ratio=vol, title=c["target_title"], wide_txt="~%dx" % round(ratio),
                final_txt=_rnd(vol), inner=nm_cap(bi["name"]).replace("The ", ""), outer_cap=nm_cap(bo["name"]))


def agent_director(c, f):
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    A, B = 180.0 / bi["diam"], 940.0 / bo["diam"]
    tint = np.array(bo["col"], np.float32)
    bot = tuple(float(x) for x in np.array((14, 10, 32), np.float32) * 0.82 + tint * 0.18)
    return dict(dur=22.0, hook_end=3.0, zoom_end=9.5, fill_end=18.0, fade_start=20.0, A=A, B=B, gap=1150.0 / A,
                bg_top=(4, 6, 16), bg_bot=bot, title=f["title"],
                beats=[("hook: question + pulsing ring", 0, 3), ("zoom-out reveals container", 3, 9.5),
                       ("fill: true 3D volumetric orbital packing", 9.5, 18), ("payoff: final number", 18, 22)])


def agent_engine(c, f, sb, out):
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    inner, outer = c["inner_body"], c["outer_body"]
    ratio, vol = f["ratio"], f["vol_ratio"]
    DUR, z0, z1, z2, zf = sb["dur"], sb["hook_end"], sb["zoom_end"], sb["fill_end"], sb["fade_start"]
    A, B, gap = sb["A"], sb["B"], sb["gap"]
    ri, ro = bi["diam"] / 2, bo["diam"] / 2
    ys_km = ro + gap
    
    # 3D Volumetric Engine instance
    vol_3d = VolumetricSphere3D(inner, ratio, outer_px_r=ro * B)
    
    bg = make_bg(sb["bg_top"], sb["bg_bot"], seed=11)
    tf, tl = wrap_fit(sb["title"], 940, 3, 92)
    inner_nm, outer_nm = f["inner"], f["outer_cap"]
    lab_fill = "%s fit inside %s" % (bi["plural"], nm_mid(bo["name"]))

    aud = Audio(DUR)
    aud.add(0.0, ambient_sub_bass(DUR, f0=55.0, amp=0.30))
    aud.add(0.2, whoosh(0.9, 400, 2200, 0.35, 5))
    aud.add(z0 - 0.1, whoosh(2.0, 150, 900, 0.5, 2))
    aud.add(z1, boom())
    for k in range(85):
        aud.add(z1 + (z2 - z1) * sm_inv(k / 85.0), tick(900 + 12 * k, 0.28))
    aud.add(z2 + 0.3, whoosh(1.4, 2200, 300, 0.45, 3))
    aud.add(z2 + 0.35, boom(1.2, 0.65))

    def frame_at(t):
        fr = bg.copy()
        cam_drift = 1.0 + (0.05 * (t / DUR))
        u = sm((t - z0) / (z1 - z0))
        sc = A * (B / A) ** u * cam_drift
        camy = ys_km * u
        sy = CY + (ys_km - camy) * sc
        fe = sm((t - z1) / (z2 - z1))
        in_a = sm(t / 0.5) * (1 - sm((t - (z1 - 2.5)) / 2.0))

        # Dynamic Parallax Orbital Camera angle
        cam_orbit = 0.35 * (t - z1) if t >= z1 else 0.0

        if t >= z0 - 0.2:
            # When filling, transform the container into an atmospheric glass shell
            # so the true 3D planets are visible inside!
            is_filling = (t >= z1)
            draw_body(fr, outer, 540, sy, ro * sc, 1.0, fe, spin=0.35 * t, t=t, glass_mode=is_filling)
            
            # Draw true 3D volumetric spheres inside!
            vol_3d.draw(fr, fe, alpha=1.0, center_x=540, center_y=sy, outer_px_r=ro * sc * 0.90, cam_orbit=cam_orbit)

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

        fr = bloom(fr, 0.45, 155.0, 8)
        sa = 1 - sm((t - zf) / (DUR - zf))
        if sa < 1:
            fr = bg * (1 - sa) + fr * sa

        tx = Text()
        tx.d.rounded_rectangle([60, 100, 1020, 350], 25, fill=(0, 0, 0, int(120 * sm(t / 0.35))))
        tx.lines(W / 2, 225, tl, tf, a=sm(t / 0.35))

        if t < z1 - 1.7:
            tx.put((540, iy + ri * sc + 64), inner_nm, font(46), (160, 215, 255),
                   (1 - sm((t - (z1 - 3.0)) / 1.0)) * sm((t - 0.4) / 0.5))

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

        tx.d.rounded_rectangle([70, 1490, 1010, 1830], 35, fill=(8, 10, 24, int(205 * ca)))
        border_col = (110, 255, 180, int(170 * ca)) if t >= z2 else (255, 220, 110, int(150 * ca))
        tx.d.rounded_rectangle([70, 1490, 1010, 1830], 35, outline=border_col, width=4)

        num_col = (255, 230, 110) if t < z2 else (110, 255, 175)
        tx.put((540, 1610), num, fit_font(num, 900, 180, 55), num_col, ca)
        tx.put((540, 1745), lab, fit_font(lab, 900, 52, 24), (240, 245, 255), ca)

        if t >= z1:
            tx.put((540, 1865), "3D volumetric simulation - Counter reflects volume", font(30), (180, 185, 215),
                   0.95 * sm((t - z1 - 0.5) / 0.6) * ca)

        return tx.apply(np.clip(fr, 0, 255).astype(np.uint8)).tobytes()

    base = os.path.join(out, c["concept_id"].lower())
    render_video(base + ".mp4", DUR, frame_at, aud)
    return base + ".mp4"


def agent_publisher(c, f, sb, mp4, ledger, ledger_path, update_ledger=True):
    quality_check(mp4, sb["dur"])
    bi, bo = BODIES[c["inner_body"]], BODIES[c["outer_body"]]
    title = c["target_title"] + " #shorts"
    desc = ("%s\nDiameter ratio %.1fx - volume ratio %s.\n3D Volumetric orbital simulation. "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["auto", "scale", "all"])
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--bodies", default="data/bodies.json")
    ap.add_argument("--ledger", default="ledger.json")
    ap.add_argument("--count", type=int, default=1)
    a = ap.parse_args()
    for t in ("ffmpeg", "ffprobe"):
        if shutil.which(t) is None:
            fail("%s not found on PATH" % t, 3)
    os.makedirs(a.out, exist_ok=True)
    load_bodies(a.bodies)
    print("bodies loaded: %d" % len(BODIES))
    for k in range(a.count):
        t0 = time.time()
        path = run_pipeline(a.out, a.ledger, None if a.seed < 0 else a.seed + k)
        if path:
            print("done -> %s  (%.0fs)\n" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
