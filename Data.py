"""Veri katmani: yerel dogrulanmis DB (hizli, guvenilir) + Gemini (bilinmeyen nesneler).
Her deger kaynak notuyla birlikte tasinir ve data_cache.json'a yazilir."""
import json, os, re, urllib.request, urllib.error

CACHE_PATH = os.environ.get("DATA_CACHE", "/tmp/data_cache.json")
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-flash-latest",
                             "gemini-2.5-flash", "gemini-2.5-flash-lite"] if m]
RSUN = 695700.0

# Hacimsel ortalama yaricaplar (km) - NASA planetary fact sheet / IAU nominal degerler
LOCAL = {
    "sun":     dict(name="Sun", radius_km=RSUN, kind="star", tint=(1.0, 1.0, 1.0), note="IAU nominal solar radius"),
    "mercury": dict(name="Mercury", radius_km=2439.4, kind="mercury", note="NASA fact sheet (volumetric mean)"),
    "venus":   dict(name="Venus", radius_km=6051.8, kind="venus", note="NASA fact sheet (volumetric mean)"),
    "earth":   dict(name="Earth", radius_km=6371.0, kind="earth", note="NASA fact sheet (volumetric mean)"),
    "moon":    dict(name="Moon", radius_km=1737.4, kind="moon", note="NASA fact sheet (volumetric mean)"),
    "mars":    dict(name="Mars", radius_km=3389.5, kind="mars", note="NASA fact sheet (volumetric mean)"),
    "jupiter": dict(name="Jupiter", radius_km=69911.0, kind="jupiter", note="NASA fact sheet (volumetric mean)"),
    "saturn":  dict(name="Saturn", radius_km=58232.0, kind="saturn", rings=True, note="NASA fact sheet (volumetric mean, rings excluded)"),
    "uranus":  dict(name="Uranus", radius_km=25362.0, kind="uranus", note="NASA fact sheet (volumetric mean)"),
    "neptune": dict(name="Neptune", radius_km=24622.0, kind="neptune", note="NASA fact sheet (volumetric mean)"),
    "pluto":   dict(name="Pluto", radius_km=1188.3, kind="moon", note="NASA fact sheet"),
    "proxima centauri": dict(name="Proxima Centauri", radius_km=0.1542 * RSUN, kind="star", tint=(1.0, 0.45, 0.25), note="~0.154 solar radii (literature)"),
    "sirius":  dict(name="Sirius A", radius_km=1.711 * RSUN, kind="star", tint=(0.75, 0.85, 1.0), note="~1.71 solar radii (literature)"),
    "betelgeuse": dict(name="Betelgeuse", radius_km=764 * RSUN, kind="star", tint=(1.0, 0.35, 0.15), note="~764 solar radii (Joyce+ 2020), estimates vary widely"),
}
ALIASES = {"sirius a": "sirius", "güneş": "sun", "dünya": "earth", "jüpiter": "jupiter", "satürn": "saturn",
           "ay": "moon", "mars": "mars", "venüs": "venus", "merkür": "mercury", "uranüs": "uranus", "neptün": "neptune",
           "proxima": "proxima centauri", "terra": "earth", "the sun": "sun", "the earth": "earth", "the moon": "moon"}


def _load():
    try:
        return json.load(open(CACHE_PATH))
    except Exception:
        return {}


def _save(c):
    try:
        json.dump(c, open(CACHE_PATH, "w"), indent=1)
    except Exception:
        pass


def gemini(prompt, ground=False, timeout=60):
    """Gemini REST cagrisi. GEMINI_API_KEY yoksa None doner. ground=True: Google Search ile dogrulama."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}}
    if ground:
        body["tools"] = [{"google_search": {}}]
    else:
        body["generationConfig"]["responseMimeType"] = "application/json"
    last = None
    for model in GEMINI_MODELS:
        url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
        except urllib.error.HTTPError as e:
            last = "%s %s" % (model, e.code)
            if e.code in (404, 400):
                continue  # model adi gecersiz olabilir -> siradaki
            if e.code == 429:
                print("Gemini kota doldu (429)")
                return None
        except Exception as e:
            last = str(e)
    print("Gemini basarisiz:", last)
    return None


def lookup(name):
    """Nesne bilgisi: {name, radius_km, kind, note, source}. Once yerel, sonra cache, sonra Gemini."""
    k = name.strip().lower()
    k = ALIASES.get(k, k)
    if k in LOCAL:
        d = dict(LOCAL[k]); d["source"] = "local"
        return d
    cache = _load()
    if k in cache:
        return cache[k]
    prompt = ("Give the mean (volumetric) radius in kilometers of the astronomical object or celestial body '%s'. "
              "Use the most recent reliable scientific value. Reply ONLY JSON: "
              '{"name": str, "radius_km": number, "type": "star|rocky|gas_giant|ice_giant|moon", '
              '"star_color_rgb": [r,g,b 0-1] or null, "note": "source and uncertainty in <=12 words"}' % name)
    d = gemini(prompt, ground=True)
    if not d or not d.get("radius_km"):
        raise ValueError("'%s' icin veri bulunamadi (yerel DB'de yok, Gemini yanit vermedi)." % name)
    t = d.get("type", "rocky")
    kind = {"star": "star", "rocky": "rock", "gas_giant": "gas", "ice_giant": "neptune", "moon": "moon"}.get(t, "rock")
    out = dict(name=d.get("name", name), radius_km=float(d["radius_km"]), kind=kind, note=d.get("note", "Gemini + Google Search"),
               source="gemini")
    if kind == "star":
        out["tint"] = tuple(d.get("star_color_rgb") or (1, 0.8, 0.5))
    cache[k] = out
    _save(cache)
    return out


def parse_topic(topic):
    """'How many Earths fit inside the Sun?' -> (unit_name, container_name). Once regex, sonra Gemini."""
    t = topic.strip().rstrip("?.! ")
    m = re.search(r"how many\s+(.+?)\s+(?:can\s+)?fit(?:s)?\s+(?:in|inside|into|within)\s+(?:the\s+)?(.+)$", t, re.I)
    if m:
        u = re.sub(r"s$", "", m.group(1).strip(), flags=re.I) if m.group(1).lower().endswith("s") and m.group(1).lower() not in ("mars",) else m.group(1)
        return u.strip(), m.group(2).strip()
    m = re.search(r"(.+?)\s+(?:vs\.?|in|inside)\s+(.+)", t, re.I)
    d = gemini('Extract from this video topic the small object (unit) and the big container object for a '
               '"how many X fit inside Y" comparison. Topic: "%s". Reply ONLY JSON {"unit": str, "container": str}' % topic)
    if d and d.get("unit") and d.get("container"):
        return d["unit"], d["container"]
    if m:
        return m.group(1).strip(), m.group(2).strip()
    raise ValueError("Konu anlasilamadi. Ornek: 'How many Earths fit inside the Sun?'")
