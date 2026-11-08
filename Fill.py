"""KIYAS / DOLDURMA modulu: 'How many X fit inside Y?'
Zaman cizelgesi: A tanitim -> B kozmik zoom-out -> C doldurma+sayac -> D final."""
import math, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .bodies import render_body, composite, make_starfield

FONT_BOLD = None
for _p in ["/usr/share/fonts/opentype/inter/Inter-Bold.otf", "/usr/share/fonts/truetype/inter/Inter-Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
    try:
        ImageFont.truetype(_p, 20)
        FONT_BOLD = _p
        break
    except Exception:
        continue

GOLD = (1.0, 0.87, 0.35)
GREEN = (0.45, 1.0, 0.72)
WHITE = (1, 1, 1)
GREY = (0.62, 0.66, 0.74)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_sine(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def fmt_int(n):
    return "{:,}".format(int(n))


def fmt_big(n):
    if n >= 1e12: return "%.1f TRILLION" % (n / 1e12)
    if n >= 1e9: return "%.1f BILLION" % (n / 1e9)
    if n >= 1e6: return "%.1f MILLION" % (n / 1e6)
    if n >= 1e5: return "%d THOUSAND" % int(round(n / 1e3))
    return fmt_int(round(n, -1) if n >= 1000 else round(n))


def plural(name):
    return name if name.lower().endswith("s") else name + "s"


def article(name):
    return "the " + name if name.lower() in ("sun", "moon", "milky way") else name


@functools.lru_cache(maxsize=1024)
def text_mask(text, size):
    font = ImageFont.truetype(FONT_BOLD, size)
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 12, b - t + 12
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((6 - l, 6 - t), text, font=font, fill=255)
    return np.asarray(im, np.float32) / 255.0


def blit_text(dst, text, size, cx, cy, color, alpha=1.0, shadow=0.55):
    if alpha <= 0.001:
        return
    m = text_mask(text, int(size))
    h, w = m.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = dst.shape[:2]
    def put(mm, ox, oy, col, a):
        xs0, ys0 = max(0, x0 + ox), max(0, y0 + oy)
        xs1, ys1 = min(W, x0 + ox + w), min(H, y0 + oy + h)
        if xs1 <= xs0 or ys1 <= ys0:
            return
        sub = mm[ys0 - (y0 + oy):ys1 - (y0 + oy), xs0 - (x0 + ox):xs1 - (x0 + ox)] * a
        reg = dst[ys0:ys1, xs0:xs1]
        reg[:] = reg * (1 - sub[..., None]) + np.array(col, np.float32) * sub[..., None]
    if shadow:
        o = max(2, int(size * 0.04))
        put(m, o, o, (0, 0, 0), shadow * alpha)
    put(m, 0, 0, color, alpha)


def fit_size(text, size, maxw):
    while size > 20 and text_mask(text, size).shape[1] > maxw:
        size -= 4
    return size


def wrap(text, size, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if text_mask(t, size).shape[1] <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


@functools.lru_cache(maxsize=16)
def _panel_masks(w, h, r, bw):
    ss = 3
    big = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(big).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    outer = np.asarray(big.resize((w, h), Image.LANCZOS), np.float32) / 255
    inner_i = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(inner_i).rounded_rectangle((bw * ss, bw * ss, (w - bw) * ss - 1, (h - bw) * ss - 1), max(r - bw, 1) * ss, fill=255)
    inner = np.asarray(inner_i.resize((w, h), Image.LANCZOS), np.float32) / 255
    return outer, np.clip(outer - inner, 0, 1)


def panel(dst, x0, y0, x1, y1, border, alpha=0.78, r=40, bw=3):
    outer, edge = _panel_masks(int(x1 - x0), int(y1 - y0), int(r), int(bw))
    reg = dst[int(y0):int(y1), int(x0):int(x1)]
    a = outer * alpha
    reg[:] = reg * (1 - a[..., None]) + np.array([0.02, 0.03, 0.08], np.float32) * a[..., None]
    e = edge * 0.9
    reg[:] = reg * (1 - e[..., None]) + np.array(border, np.float32) * e[..., None]


class FillScene:
    A, B, C, D = 4.0, 5.0, 14.0, 4.0

    def __init__(self, unit, cont, topic, scale=1.0, fps=30, mode="volume", seed=3):
        self.unit, self.cont, self.topic, self.fps = unit, cont, topic, fps
        self.S = scale
        self.W, self.H = int(round(1080 * scale)), int(round(1920 * scale))
        self.T = self.A + self.B + self.C + self.D
        self.t_B, self.t_C, self.t_D = self.A, self.A + self.B, self.A + self.B + self.C
        self.ratio_d = cont["radius_km"] / unit["radius_km"]
        self.packed = (mode == "packed")
        self.total = self.ratio_d ** 3 * (0.64 if self.packed else 1.0)
        S = self.S
        self.CX, self.CY, self.RC, self.RU_A = 540 * S, 1000 * S, 430 * S, 275 * S
        self.rs = np.random.RandomState(seed)
        self.canvas = (self.W, self.H)
        pad = 8
        self.bg = make_starfield(self.W + pad * 8, self.H + pad * 8, seed=5)
        self._build_container()
        self._build_fill()

    # ---------------------------------------------------------------- hazirlik
    def _build_container(self):
        spin_c = 0.05 * self.t_C
        self.spin_c_final = spin_c
        self.cont_layer = render_body(self.cont, self.CX, self.CY, self.RC, self.canvas,
                                      spin=spin_c, elev=0.30, roll=-0.30)

    def _sprite_tiles(self, D):
        size = int(math.ceil(D)) + 2
        unit = dict(self.unit); unit["rings"] = False
        tiles = []
        for v in range(6):
            if D >= 26:
                R = D / 2 * 0.985
                lay = render_body(unit, size / 2, size / 2, R, (size, size), spin=v * 1.1, elev=0.3, roll=-0.3, glow=False)
                rgb, a, x0, y0 = lay
                T = np.zeros((size, size, 4), np.float32)
                T[y0:y0 + a.shape[0], x0:x0 + a.shape[1], :3] = rgb
                T[y0:y0 + a.shape[0], x0:x0 + a.shape[1], 3] = a
            else:
                SSZ = 128
                lay = render_body(unit, SSZ / 2, SSZ / 2, SSZ / 2 * 0.985, (SSZ, SSZ), spin=v * 1.1, elev=0.3, roll=-0.3, glow=False)
                rgb, a, x0, y0 = lay
                big = np.zeros((SSZ, SSZ, 4), np.float32)
                big[y0:y0 + a.shape[0], x0:x0 + a.shape[1], :3] = rgb
                big[y0:y0 + a.shape[0], x0:x0 + a.shape[1], 3] = a
                chans = []
                for c in range(4):
                    im = Image.fromarray(big[..., c], mode="F").resize((int(round(D)) if False else size - 2, size - 2), Image.LANCZOS)
                    chans.append(np.asarray(im, np.float32))
                small = np.stack(chans, -1)
                T = np.zeros((size, size, 4), np.float32)
                T[1:size - 1, 1:size - 1] = small
            tiles.append(T)
        return tiles, size

    def _build_fill(self):
        S = self.S
        D = 2 * self.RC / self.ratio_d
        self.sprites_enlarged = D < 3.0
        D = max(D, 3.0)
        self.D_fill = D
        tiles, size = self._sprite_tiles(D)
        d = D * 1.03
        dy = d * math.sqrt(3) / 2
        J = int(self.RC / dy) + 2
        I = int(self.RC / d) + 2
        pts = []
        for j in range(-J, J + 1):
            xo = d / 2 if (j % 2) else 0.0
            y = self.CY + j * dy
            for i in range(-I, I + 1):
                x = self.CX + xo + i * d
                if math.hypot(x - self.CX, y - self.CY) <= self.RC - D / 2 - 1:
                    pts.append((x, y))
        pts = np.array(pts)
        self.n2d = len(pts)
        key = -pts[:, 1] + self.rs.normal(0, 0.55 * d, len(pts))   # alttan uste
        key[np.argmin(np.hypot(pts[:, 0] - self.CX, pts[:, 1] - self.CY))] = -1e9  # merkez ilk
        order = np.argsort(key)
        rank = np.empty(len(pts)); rank[order] = np.arange(len(pts)) / max(len(pts) - 1, 1)
        bx0, bx1 = int(self.CX - self.RC) - 2, int(self.CX + self.RC) + 3
        by0, by1 = int(self.CY - self.RC) - 2, int(self.CY + self.RC) + 3
        self.bbox = (bx0, by0, bx1, by1)
        Wb, Hb = bx1 - bx0, by1 - by0
        layer = np.zeros((Hb, Wb, 4), np.float32)
        arr = np.full((Hb, Wb), 9.0, np.float32)
        for k, (x, y) in enumerate(pts):
            T = tiles[self.rs.randint(len(tiles))]
            px, py = int(round(x - size / 2)) - bx0, int(round(y - size / 2)) - by0
            if px < 0 or py < 0 or px + size > Wb or py + size > Hb:
                continue
            sl = layer[py:py + size, px:px + size]
            sl[:] = T + sl * (1 - T[..., 3:4])
            a_sl = arr[py:py + size, px:px + size]
            a_sl[T[..., 3] > 0.04] = rank[k] if rank[k] > 0 else -1.0
        self.fill_rgb = layer[..., :3]
        self.fill_a = layer[..., 3]
        self.fill_arr = arr

    # ---------------------------------------------------------------- yardimcilar
    def progress(self, t):
        return ease_sine((t - (self.t_C + 0.6)) / (self.C - 1.2))

    def sfx_events(self):
        ticks, last = [], 0
        for i in range(int(self.C * self.fps)):
            t = self.t_C + i / self.fps
            k = int(self.progress(t) * 70)
            if k > last:
                ticks.append((t, self.progress(t)))
                last = k
        return dict(whoosh=self.t_B, ticks=ticks, ding=self.t_D + 0.1)

    def _bg(self, t):
        ox, oy = int(t * 3) % 60, int(t * 2) % 60
        return self.bg[oy:oy + self.H, ox:ox + self.W].copy()

    def _cont_half(self, D_c, spin, opacity, dst):
        """Container'i yari cozunurlukte render edip buyutur (zoom-out sahnesi icin)."""
        h = 0.5
        lay = render_body(self.cont, self.CX * h, self.CY * h, D_c / 2 * h, (int(self.W * h), int(self.H * h)),
                          spin=spin, elev=0.30, roll=-0.30)
        if lay is None:
            return
        rgb, a, x0, y0 = lay
        hh, ww = a.shape
        chans = [np.asarray(Image.fromarray(np.clip(rgb[..., c], 0, 1), mode="F").resize((int(ww / h), int(hh / h)), Image.BILINEAR), np.float32) for c in range(3)]
        al = np.asarray(Image.fromarray(a, mode="F").resize((int(ww / h), int(hh / h)), Image.BILINEAR), np.float32)
        composite(dst, (np.stack(chans, -1), al, int(x0 / h), int(y0 / h)), opacity)

    # ---------------------------------------------------------------- KARE
    def frame(self, t):
        S, W, H = self.S, self.W, self.H
        dst = self._bg(t)
        unit, cont = self.unit, self.cont

        if t < self.t_B:                                            # A: tanitim
            zoom = 1 + 0.05 * (t / self.A)
            R = self.RU_A * zoom
            composite(dst, render_body(unit, self.CX, self.CY, R, self.canvas, spin=0.5 * t, elev=0.30, roll=-0.30))
            stage = "A"
        elif t < self.t_C:                                          # B: zoom-out
            e = ease((t - self.t_B) / self.B)
            D0 = 2 * self.RU_A * self.ratio_d
            D1 = 2 * self.RC
            D_c = math.exp(math.log(D0) + (math.log(D1) - math.log(D0)) * e)
            op = min(1.0, (t - self.t_B) / 1.2)
            if D_c > 1500 * S:
                self._cont_half(D_c, 0.05 * t, ease(op), dst)
            else:
                composite(dst, render_body(cont, self.CX, self.CY, D_c / 2, self.canvas, spin=0.05 * t, elev=0.30, roll=-0.30), ease(op))
            R_u = D_c / self.ratio_d / 2
            if R_u > 0.6:
                composite(dst, render_body(unit, self.CX, self.CY, R_u, self.canvas, spin=0.5 * t, elev=0.30, roll=-0.30))
            stage = "B"
        else:                                                       # C/D: doldurma
            p = self.progress(t)
            dim = 1.0 - 0.38 * ease((t - self.t_C) / 1.0)
            rgb, a, x0, y0 = self.cont_layer
            composite(dst, (rgb * dim, a, x0, y0))
            bx0, by0, bx1, by1 = self.bbox
            m = np.clip((p - self.fill_arr) / 0.006, 0, 1)
            reg = dst[by0:by1, bx0:bx1]
            reg[:] = self.fill_rgb * m[..., None] + reg * (1 - self.fill_a * m)[..., None]
            stage = "C" if t < self.t_D else "D"

        # ---- ust baslik
        grad = np.clip(1 - np.arange(H) / (420 * S), 0, 1) ** 1.5 * 0.65
        dst *= (1 - grad)[:, None, None]
        fa = ease(t / 0.8)
        title = self.topic if self.topic.endswith("?") else self.topic + "?"
        size = int(80 * S)
        lines = wrap(title, size, int(960 * S))
        y = 215 * S - (len(lines) - 1) * size * 0.58
        for ln in lines:
            blit_text(dst, ln, size, W / 2, y, WHITE, alpha=fa)
            y += size * 1.16

        # ---- alt panel
        un, cn = self.unit["name"], self.cont["name"]
        final = stage == "D"
        panel(dst, 60 * S, 1500 * S, 1020 * S, 1800 * S, GREEN if final else (0.85, 0.68, 0.25))
        pcx = W / 2
        if stage == "A":
            a_ = ease((t - 0.3) / 0.6)
            blit_text(dst, un.upper(), 120 * S, pcx, 1612 * S, WHITE, a_)
            blit_text(dst, "Radius: %s km" % fmt_int(round(unit["radius_km"])), 44 * S, pcx, 1735 * S, GREY, a_)
        elif stage == "B":
            a_ = ease((t - self.t_B) / 0.8)
            r = self.ratio_d
            big = ("{:,}\u00d7".format(int(round(r)))) if r >= 10 else ("%.1f\u00d7" % r)
            blit_text(dst, big, int(fit_size(big, int(150 * S), int(860 * S))), pcx, 1620 * S, GOLD, a_)
            blit_text(dst, "%s is wider than %s" % (cn, un), 44 * S, pcx, 1735 * S, WHITE, a_)
        else:
            p = self.progress(t)
            if final:
                txt, col = "\u2248 " + fmt_big(self.total), GREEN
            else:
                txt, col = fmt_int(max(0, int(p * self.total))), GOLD
            sz = fit_size(txt, int(150 * S), int(880 * S))
            blit_text(dst, txt, sz, pcx, 1620 * S, col)
            blit_text(dst, "%s fit inside %s" % (plural(un), article(cn)), 46 * S, pcx, 1738 * S, WHITE)
        note = "2D slice visualized \u2022 counter reflects 3D %s" % ("volume (random packing 64%)" if self.packed else "volume ratio")
        blit_text(dst, note, 27 * S, pcx, 1843 * S, GREY, 0.9, shadow=0.0)
        stars = [b for b in (self.unit, self.cont) if b["kind"] == "star" and b["name"] != "Sun"]
        if self.unit.get("source") == "gemini" or self.cont.get("source") == "gemini":
            src = "Radii: web sources via Gemini (approximate)"
        elif stars:
            src = "Radii: NASA / literature \u2022 stellar radii are estimates"
        else:
            src = "Radii: NASA fact sheets"
        blit_text(dst, src, 22 * S, pcx, 1876 * S, (0.45, 0.5, 0.6), 0.9, shadow=0.0)

        # ---- giris/cikis
        f = min(1.0, t / 0.5) * min(1.0, (self.T - t) / 0.6)
        dst *= f
        return (np.clip(dst, 0, 1) * 255 + 0.5).astype(np.uint8)
