"""Modal giris noktasi. Kullanim:
  modal run modal_render.py --topic "How many Earths fit inside the Sun?" --preview
  modal run modal_render.py --topic "How many Earths fit inside Jupiter?"
GPU gerekmez: render CPU'da (4 cekirdek) yapilir, video basina birkac sent.
"""
import os
import modal

app = modal.App("shorts-compare-engine")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "fonts-dejavu-core")
    .pip_install("numpy", "scipy", "pillow")
    .add_local_python_source("engine")
)
if os.path.isdir("assets"):          # assets/music/*.mp3 (telifsiz muzikler) varsa imaja ekle
    image = image.add_local_dir("assets", "/root/assets")

# GEMINI_API_KEY, GitHub secret'indan ortam degiskeni olarak gelir (yoksa yerel veri tabani kullanilir)
secrets = [modal.Secret.from_dict({"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")})]


@app.function(image=image, cpu=4.0, memory=4096, timeout=1500, secrets=secrets)
def render(topic: str, preview: bool = False, mode: str = "volume") -> bytes:
    from engine.make import make_video, make_preview
    if preview:
        out = make_preview(topic, "/tmp/out.png", scale=0.5)
    else:
        out = make_video(topic, "/tmp/out.mp4", scale=1.0, workers=4,
                         music_dir="/root/assets/music", mode=mode)
    with open(out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(topic: str = "How many Earths fit inside the Sun?", preview: bool = False, mode: str = "volume"):
    data = render.remote(topic, preview, mode)
    name = "preview.png" if preview else "final_video.mp4"
    with open(name, "wb") as f:
        f.write(data)
    print("Kaydedildi: %s (%d KB)" % (name, len(data) // 1024))
