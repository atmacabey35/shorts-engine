"""Gercekci kure/gezegen render'i (numpy, GPU gerekmez).
Dokular proseduerel uretilir, isik/atmosfer/halka hesabi fizige yakin."""
import math
import numpy as np
from PIL import Image
from .noise import fbm3, norm01, smoothstep

MAP_W, MAP_H = 1024, 512
_map_cache = {}


def _lonlat(W=MAP_W, H=MAP_H):
    lon = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    p = np.stack([np.cos(LAT) * np.cos(LON), np.sin(LAT), np.cos(LAT) * np.sin(LON)], -1)
    return LON.astype(np.float32), LAT.astype(np.float32), p.astype(np.float32)


def _pal(t, stops):
    """t:[0,1] -> renk, stops: [(pos,(r,g,b)),...]"""
    pos = np.array([s[0] for s in stops])
    out = np.zeros(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, [s[1][c] for s in stops])
    return out


# ------------------------------------------------------------------ DOKULAR
def _map_earth():
    LON, LAT, p = _lonlat(2048, 1024)
    h = fbm3(p, 3, 6, seed=14)
    sea = np.percentile(h, 70)
    e = h - sea
    land = smoothstep(-0.003, 0.003, e)
    moist = norm01(fbm3(p, 3, 4, seed=40))
    alat = np.abs(LAT)
    dry = np.clip(1.7 * (0.5 - moist) + 0.7 * np.exp(-((alat - 0.5) / 0.28) ** 2), 0, 1)
    green = np.array([0.07, 0.26, 0.07], np.float32) + 0.07 * (moist[..., None] * np.array([0.5, 1, 0.3], np.float32))
    sand = np.array([0.68, 0.54, 0.31], np.float32)
    lc = green * (1 - dry[..., None]) + sand * dry[..., None]
    elev = np.clip(e / max(np.percentile(e[e > 0], 92), 1e-6), 0, 1)
    mount = np.array([0.40, 0.35, 0.30], np.float32)
    lc = lc * (1 - 0.7 * elev[..., None] ** 1.5) + mount * 0.7 * elev[..., None] ** 1.5
    ice = smoothstep(1.12, 1.32, alat + 0.1 * (norm01(fbm3(p, 6, 3, seed=5)) - 0.5)) + 0.6 * smoothstep(0.85, 1.0, elev) * 0
    ice = np.clip(ice, 0, 1)
    lc = lc * (1 - ice[..., None]) + np.array([0.93, 0.95, 0.98], np.float32) * ice[..., None]
    depth = np.clip(-e / 0.10, 0, 1)
    oc = np.array([0.07, 0.34, 0.52], np.float32) * (1 - depth[..., None]) + np.array([0.008, 0.045, 0.2], np.float32) * depth[..., None]
    oc = oc * (1 - ice[..., None] * 0.9) + np.array([0.85, 0.9, 0.95], np.float32) * ice[..., None] * 0.9
    rgb = lc * land[..., None] + oc * (1 - land[..., None])
    cl = norm01(fbm3(p * np.array([1, 1.7, 1], np.float32), 2.5, 6, seed=70))
    cloud = smoothstep(0.50, 0.78, cl)
    lights = land * (1 - ice) * smoothstep(0.62, 0.8, norm01(fbm3(p, 14, 3, seed=90))) * smoothstep(0.35, 0.55, norm01(fbm3(p, 4, 3, seed=91)))
    return dict(rgb=rgb.astype(np.float32), ocean=(1 - land * (1 - ice)).astype(np.float32),
                cloud=cloud.astype(np.float32), lights=lights.astype(np.float32))


def _map_sun(tint=(1.0, 1.0, 1.0)):
    LON, LAT, p = _lonlat(2048, 1024)
    g = norm01(fbm3(p, 5, 6, seed=3) * 0.45 + fbm3(p, 16, 3, seed=8) * 0.55)
    rgb = _pal(g, [(0, (0.85, 0.22, 0.02)), (0.5, (1.0, 0.55, 0.08)), (1.0, (1.0, 0.88, 0.45))])
    spots = smoothstep(0.90, 0.97, norm01(fbm3(p, 8, 3, seed=14))) * smoothstep(0.5, 0.25, np.abs(LAT))
    rgb *= (1 - 0.55 * spots[..., None])
    rgb *= np.array(tint, np.float32)
    return dict(rgb=np.clip(rgb, 0, 1.4).astype(np.float32))


def _map_jupiter():
    LON, LAT, p = _lonlat()
    warp = LAT + 0.05 * (fbm3(p, 3, 4, seed=21) - 0.5) + 0.02 * (fbm3(p, 10, 3, seed=22) - 0.5)
    b = 0.55 * np.sin(warp * 21) + 0.3 * np.sin(warp * 45 + 1.0) + 0.15 * np.sin(warp * 90 + 2.0)
    t = np.clip((b + 1) / 2 + 0.10 * (fbm3(p, 14, 3, seed=23) - 0.5), 0, 1)
    rgb = _pal(t, [(0, (0.46, 0.28, 0.2)), (0.3, (0.66, 0.45, 0.31)), (0.6, (0.86, 0.74, 0.58)), (1, (0.95, 0.91, 0.84))])
    dx = np.arctan2(np.sin(LON - 0.9), np.cos(LON - 0.9)) / 0.20
    dy = (LAT + 0.40) / 0.085
    g = np.exp(-(dx ** 2 + dy ** 2))
    rim = np.exp(-((np.sqrt(dx ** 2 + dy ** 2) - 1.4) / 0.4) ** 2)
    rgb = rgb * (1 - 0.8 * g[..., None]) + np.array([0.72, 0.3, 0.18], np.float32) * 0.8 * g[..., None]
    rgb = rgb * (1 - 0.25 * rim[..., None])
    return dict(rgb=rgb.astype(np.float32))


def _map_saturn():
    LON, LAT, p = _lonlat()
    warp = LAT + 0.012 * (fbm3(p, 3, 3, seed=31) - 0.5)
    b = 0.6 * np.sin(warp * 16) + 0.3 * np.sin(warp * 37 + 0.5) + 0.1 * np.sin(warp * 80)
    t = np.clip((b + 1) / 2 + 0.05 * (fbm3(p, 10, 3, seed=32) - 0.5), 0, 1)
    rgb = _pal(t, [(0, (0.72, 0.60, 0.40)), (0.5, (0.86, 0.75, 0.52)), (1, (0.95, 0.87, 0.66))])
    polar = smoothstep(0.9, 1.45, np.abs(LAT))
    rgb = rgb * (1 - 0.25 * polar[..., None]) + np.array([0.55, 0.6, 0.62], np.float32) * 0.25 * polar[..., None]
    return dict(rgb=rgb.astype(np.float32))


def _map_mars():
    LON, LAT, p = _lonlat()
    g = norm01(fbm3(p, 3, 6, seed=41))
    rgb = _pal(g, [(0, (0.30, 0.15, 0.09)), (0.45, (0.62, 0.30, 0.16)), (1, (0.80, 0.50, 0.28))])
    cap = smoothstep(1.28, 1.40, np.abs(LAT) + 0.05 * fbm3(p, 8, 2, seed=2))
    rgb = rgb * (1 - cap[..., None]) + np.array([0.95, 0.95, 0.97], np.float32) * cap[..., None]
    return dict(rgb=rgb.astype(np.float32))


def _map_rock(base=(0.55, 0.54, 0.52), contrast=0.35, seed=51, maria=True):
    LON, LAT, p = _lonlat()
    g = norm01(fbm3(p, 4, 6, seed=seed))
    fine = norm01(fbm3(p, 24, 3, seed=seed + 9))
    v = 0.55 + contrast * (g - 0.5) + 0.2 * (fine - 0.5)
    if maria:
        m = smoothstep(0.55, 0.75, norm01(fbm3(p, 2, 3, seed=seed + 20)))
        v = v * (1 - 0.35 * m)
    rgb = v[..., None] * np.array(base, np.float32) / 0.55
    return dict(rgb=np.clip(rgb, 0, 1).astype(np.float32))


def _map_venus():
    LON, LAT, p = _lonlat()
    g = norm01(fbm3(p * np.array([1, 2.5, 1], np.float32), 2.5, 6, seed=61))
    rgb = _pal(g, [(0, (0.78, 0.60, 0.33)), (0.5, (0.90, 0.78, 0.52)), (1, (0.97, 0.91, 0.72))])
    return dict(rgb=rgb.astype(np.float32))


def _map_ice(c1, c2, seed, bands=9.0, dark_spot=False):
    LON, LAT, p = _lonlat()
    b = 0.5 + 0.5 * np.sin(LAT * bands + 2 * fbm3(p, 3, 3, seed=seed))
    t = np.clip(b * 0.6 + 0.4 * norm01(fbm3(p, 5, 4, seed=seed + 1)), 0, 1)
    rgb = _pal(t, [(0, c1), (1, c2)])
    if dark_spot:
        dx = np.arctan2(np.sin(LON - 1.5), np.cos(LON - 1.5)) / 0.25
        dy = (LAT + 0.35) / 0.12
        g = np.exp(-(dx ** 2 + dy ** 2))
        rgb *= (1 - 0.45 * g[..., None])
    return dict(rgb=rgb.astype(np.float32))


def get_maps(body):
    kind = body["kind"]
    key = (kind, tuple(body.get("tint", ())), body.get("seed", 0))
    if key in _map_cache:
        return _map_cache[key]
    if kind == "earth":
        m = _map_earth()
    elif kind == "star":
        m = _map_sun(body.get("tint", (1, 1, 1)))
    elif kind == "jupiter":
        m = _map_jupiter()
    elif kind == "saturn":
        m = _map_saturn()
    elif kind == "mars":
        m = _map_mars()
    elif kind == "moon":
        m = _map_rock((0.62, 0.61, 0.60), 0.40, 51)
    elif kind == "mercury":
        m = _map_rock((0.50, 0.46, 0.43), 0.45, 53, maria=False)
    elif kind == "venus":
        m = _map_venus()
    elif kind == "uranus":
        m = _map_ice((0.50, 0.80, 0.85), (0.62, 0.90, 0.93), 71, 7.0)
    elif kind == "neptune":
        m = _map_ice((0.12, 0.25, 0.75), (0.28, 0.48, 0.92), 75, 11.0, True)
    elif kind == "gas":
        m = _map_jupiter()
    else:  # rock / generic
        m = _map_rock((0.58, 0.5, 0.44), 0.45, 57 + body.get("seed", 0))
    _map_cache[key] = m
    return m


# ------------------------------------------------------------------ ORNEKLEME
def _sample(m, lon, lat):
    H, W = m.shape[:2]
    u = ((lon / (2 * np.pi) + 0.5) % 1.0) * W - 0.5
    v = np.clip((0.5 - lat / np.pi) * H - 0.5, 0, H - 1.001)
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    fx = (u - x0).astype(np.float32)
    fy = (v - y0).astype(np.float32)
    x0 %= W
    x1 = (x0 + 1) % W
    y1 = np.minimum(y0 + 1, H - 1)
    if m.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    top = m[y0, x0] * (1 - fx) + m[y0, x1] * fx
    bot = m[y1, x0] * (1 - fx) + m[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _ring_profile(rho):
    """Satürn halkasi: (opaklik, renk). rho = gezegen yaricapi biriminde."""
    a = np.zeros_like(rho)
    pts = [(1.11, 0.0), (1.24, 0.03), (1.30, 0.12), (1.45, 0.22), (1.53, 0.30),
           (1.56, 0.80), (1.75, 0.92), (1.95, 0.88), (1.96, 0.08), (2.02, 0.04),
           (2.03, 0.55), (2.20, 0.62), (2.215, 0.08), (2.23, 0.55), (2.27, 0.5), (2.28, 0.0)]
    a = np.interp(rho, [q[0] for q in pts], [q[1] for q in pts], left=0, right=0)
    fine = 0.5 + 0.5 * (np.sin(rho * 310) * 0.5 + np.sin(rho * 137 + 1.3) * 0.3 + np.sin(rho * 61) * 0.2)
    a = np.clip(a * (0.72 + 0.28 * fine), 0, 1)
    t = np.clip((rho - 1.2) / 1.1, 0, 1)
    col = np.stack([0.60 + 0.30 * np.sin(t * 2.2) ** 2 + 0.0 * t,
                    0.54 + 0.24 * np.sin(t * 2.2) ** 2,
                    0.44 + 0.18 * np.sin(t * 2.2) ** 2], -1).astype(np.float32)
    col = col * (0.85 + 0.15 * fine[..., None])
    return a.astype(np.float32), col


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ ANA RENDER
def render_body(body, cx, cy, R, canvas, spin=0.0, elev=0.35, roll=-0.35,
                L=(-0.62, 0.45, 0.64), glow=True):
    """Premultiplied RGB + alpha dondurur: (rgb_pm[h,w,3], a[h,w], x0, y0)
    Kure canvas=(W,H) icinde (cx,cy) merkezli, R piksel yaricapli."""
    CW, CH = canvas
    kind = body["kind"]
    ringed = bool(body.get("rings"))
    emissive = kind == "star"
    if ringed:
        pad = 2.45
    elif emissive:
        pad = 2.6 if glow else 1.05
    elif kind == "earth":
        pad = 1.28 if glow else 1.05
    else:
        pad = 1.06
    x0 = int(max(0, math.floor(cx - R * pad)))
    x1 = int(min(CW, math.ceil(cx + R * pad)))
    y0 = int(max(0, math.floor(cy - R * pad)))
    y1 = int(min(CH, math.ceil(cy + R * pad)))
    if x1 <= x0 or y1 <= y0:
        return None
    xs = (np.arange(x0, x1, dtype=np.float32) + 0.5 - cx) / R
    ys = -(np.arange(y0, y1, dtype=np.float32) + 0.5 - cy) / R
    X, Y = np.meshgrid(xs, ys)
    h, w = X.shape
    rr2 = X * X + Y * Y
    rr = np.sqrt(rr2)
    Ld = _unit(L)
    maps = get_maps(body)

    # donme ekseni (view uzayi): elev = gozlemcinin enlemi
    n_base = np.array([0.0, math.cos(elev), math.sin(elev)], np.float32)
    cr, sr = math.cos(roll), math.sin(roll)
    n_ax = np.array([-sr * n_base[1], cr * n_base[1], n_base[2]], np.float32)
    u_ax = np.array([cr, sr, 0.0], np.float32)
    w_ax = np.cross(n_ax, u_ax).astype(np.float32)

    # --- kure
    inside = rr2 < 1.06
    ii = np.nonzero(inside)
    Xi, Yi = X[ii], Y[ii]
    Zi = np.sqrt(np.clip(1 - Xi * Xi - Yi * Yi, 0, 1))
    N = np.stack([Xi, Yi, Zi], -1)
    xb, yb, zb = N @ u_ax, N @ n_ax, N @ w_ax
    lon = np.arctan2(zb, xb) + spin
    lat = np.arcsin(np.clip(yb, -1, 1))
    alb = _sample(maps["rgb"], lon, lat)
    ndl = N @ Ld
    sphere_alpha = np.clip((1 - np.sqrt(Xi * Xi + Yi * Yi)) * R + 0.5, 0, 1)

    if emissive:
        limb = 0.30 + 0.70 * Zi ** 0.42
        col = alb * limb[:, None] * 1.12
    else:
        diff = np.clip((ndl + 0.04) / 1.04, 0, 1)
        shade = diff ** 0.9
        # halka golgesi (kure uzerine)
        if ringed:
            nd = float(n_ax @ Ld)
            S = N
            tt = -(S @ n_ax) / (nd if abs(nd) > 1e-3 else 1e-3)
            Q = S + tt[:, None] * Ld[None, :]
            rho_q = np.linalg.norm(Q, axis=1)
            ra, _ = _ring_profile(rho_q)
            ra = np.where(tt > 0, ra, 0.0)
            shade = shade * (1 - 0.85 * ra)
        amb = 0.022
        col = alb * (amb + (1 - amb) * shade)[:, None]
        if kind == "earth":
            night = smoothstep(0.12, -0.18, ndl)
            lt = _sample(maps["lights"], lon, lat) * night
            col = col + lt[:, None] * np.array([1.0, 0.72, 0.32], np.float32) * 0.9
            cl = _sample(maps["cloud"], lon + spin * 0.12 + 0.35, lat)
            ccol = np.array([0.96, 0.97, 1.0], np.float32)[None, :] * (0.03 + 0.97 * shade)[:, None]
            col = col * (1 - 0.88 * cl[:, None]) + ccol * 0.88 * cl[:, None]
            oc = _sample(maps["ocean"], lon, lat) * (1 - cl)
            Hh = _unit(Ld + np.array([0, 0, 1.0], np.float32))
            sp = np.clip(N @ Hh, 0, 1) ** 90 * oc * (ndl > 0)
            col = col + sp[:, None] * np.array([1.0, 0.95, 0.85], np.float32) * 0.9
            fres = (1 - Zi) ** 2.6
            col = col + fres[:, None] * np.array([0.25, 0.55, 1.0], np.float32) * 1.25 * np.clip(ndl + 0.35, 0, 1)[:, None]
        else:
            limb = 0.50 + 0.50 * Zi ** 0.5
            col = col * limb[:, None]

    sphere_rgb = np.zeros((h, w, 3), np.float32)
    sphere_a = np.zeros((h, w), np.float32)
    sphere_rgb[ii] = col
    sphere_a[ii] = sphere_alpha

    out_rgb = sphere_rgb * sphere_a[..., None]
    out_a = sphere_a.copy()

    # --- halka (Saturn)
    if ringed:
        nzr = n_ax[2] if abs(n_ax[2]) > 1e-3 else 1e-3
        zr = -(n_ax[0] * X + n_ax[1] * Y) / nzr
        rho = np.sqrt(X * X + Y * Y + zr * zr)
        ra, rc = _ring_profile(rho)
        # kure golgesi halka uzerinde
        Pm = np.stack([X, Y, zr], -1)
        b = Pm @ Ld
        disc = b * b - (rho * rho - 1)
        shadow = ((b < 0) & (disc > 0) & (rho > 1)).astype(np.float32)
        shadow = shadow * smoothstep(0.0, 0.04, disc)
        lit = 1 - 0.93 * shadow
        # kure on/arka
        zs = np.sqrt(np.clip(1 - rr2, 0, 1))
        sphere_hit = rr2 < 1
        front = (~sphere_hit) | (zr > zs)
        ra_eff = np.where(front, ra, 0.0)
        # halkanin kendisi aydinlik: gozlemci isik tarafinda
        rcol = rc * lit[..., None] * 1.05
        out_rgb = out_rgb * (1 - ra_eff[..., None]) + rcol * ra_eff[..., None]
        out_a = out_a * (1 - ra_eff) + ra_eff
    # --- atmosfer / korona parlamasi (additive)
    if glow:
        d = np.maximum(rr - 1.0, 0.0)
        out_side = (rr >= 1.0).astype(np.float32)
        if kind == "earth":
            dotl = (X * Ld[0] + Y * Ld[1]) / np.maximum(rr, 1e-5) / max(math.hypot(Ld[0], Ld[1]), 1e-5)
            edge = np.clip(0.3 + 0.7 * dotl, 0.04, 1.0)
            g = (np.exp(-d * 16) * 0.75) * edge * out_side
            out_rgb = out_rgb + g[..., None] * np.array([0.22, 0.5, 1.0], np.float32)
        elif emissive:
            tint = np.array(body.get("tint", (1, 1, 1)), np.float32)
            g1 = np.exp(-d * 5.5) * 0.85 * out_side
            g2 = np.exp(-d * 1.4) * 0.30 * out_side
            base = np.array([1.0, 0.55, 0.15], np.float32) * tint
            white = np.array([1.0, 0.85, 0.55], np.float32) * tint
            out_rgb = out_rgb + g1[..., None] * white + g2[..., None] * base
    return out_rgb.astype(np.float32), out_a.astype(np.float32), x0, y0


def composite(dst, layer, opacity=1.0):
    """dst: float32 (H,W,3) 0..1 , layer: render_body ciktisi."""
    if layer is None:
        return dst
    rgb, a, x0, y0 = layer
    h, w = a.shape
    sl = dst[y0:y0 + h, x0:x0 + w]
    sl[:] = rgb * opacity + sl * (1 - a * opacity)[..., None]
    return dst


# ------------------------------------------------------------------ YILDIZ ALANI
def make_starfield(W, H, seed=5):
    rs = np.random.RandomState(seed)
    img = np.zeros((H, W, 3), np.float32)
    # hafif nebula
    from scipy.ndimage import gaussian_filter, zoom
    for (c, sc, st) in [((0.10, 0.05, 0.22), 6, 0.22), ((0.02, 0.10, 0.20), 9, 0.16), ((0.18, 0.04, 0.10), 5, 0.10)]:
        n = rs.rand(H // 40 + 2, W // 40 + 2).astype(np.float32)
        n = zoom(n, (H / n.shape[0], W / n.shape[1]), order=3)[:H, :W]
        n = gaussian_filter(n, 20)
        n = np.clip((n - n.mean()) / (n.std() + 1e-6), -1, 3)
        n = np.clip(n * 0.25 + 0.1, 0, 1) * st
        img += n[..., None] * np.array(c, np.float32)
    base = np.array([0.006, 0.008, 0.02], np.float32)
    img += base
    # yildizlar
    N = int(W * H / 900)
    xs = rs.randint(0, W, N)
    ys = rs.randint(0, H, N)
    br = rs.pareto(2.2, N) * 0.12 + 0.05
    temp = rs.rand(N)
    cols = np.where(temp[:, None] < 0.2, np.array([0.7, 0.8, 1.0]),
                    np.where(temp[:, None] > 0.85, np.array([1.0, 0.8, 0.6]), np.array([1.0, 0.97, 0.92])))
    for x, y, b, c in zip(xs, ys, br, cols):
        img[y, x] += np.minimum(b, 1.0) * c
    img = gaussian_filter(img, (0.7, 0.7, 0)) * 2.2
    # birkac parlak yildiz (glow)
    bx = rs.randint(60, W - 60, 28)
    by = rs.randint(60, H - 60, 28)
    yy, xx = np.mgrid[-14:15, -14:15].astype(np.float32)
    g = np.exp(-(xx ** 2 + yy ** 2) / 5.0) + 0.25 * np.exp(-(xx ** 2 + yy ** 2) / 40.0)
    for x, y in zip(bx, by):
        s = rs.uniform(0.3, 0.9)
        c = np.array([0.85, 0.9, 1.0]) if rs.rand() < 0.6 else np.array([1.0, 0.9, 0.75])
        img[y - 14:y + 15, x - 14:x + 15] += (g * s)[..., None] * c
    return np.clip(img, 0, 1).astype(np.float32)
