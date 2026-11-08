import modal
import os

app = modal.App("shorts-blender-engine")

# Blender 3.6 LTS + bağımlılıklar
image = (
    modal.Image.debian_slim()
    .apt_install("wget", "xz-utils", "ffmpeg", "libxrender1", "libxi6", "libgl1", "libxkbcommon-x11-0", "libsm6")
    .run_commands(
        "wget -q -c https://download.blender.org/release/Blender3.6/blender-3.6.18-linux-x64.tar.xz",
        "tar -xf blender-3.6.18-linux-x64.tar.xz -C /opt",
        "ln -sf /opt/blender-3.6.18-linux-x64/blender /usr/local/bin/blender",
        "rm blender-3.6.18-linux-x64.tar.xz"
    )
)

@app.function(
    gpu="T4",
    timeout=300,
    image=image
)
def render_scene():
    import subprocess, sys

    blender_script = """
import bpy, math, random, sys

print("=== BLENDER BASLADI ===")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Dikey Shorts Ayarları (1080x1920)
scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 60  # Hızlı ilk doğrulama (2 saniye)

# Render Motoru: Cycles GPU
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.compute_device_type = 'CUDA'
cprefs.get_devices()

cuda_found = False
for d in cprefs.devices:
    d.use = True
    cuda_found = True
    print(f"CUDA Aygıtı Kullanılıyor: {d.name}")

if not cuda_found:
    print("UYARI: CUDA aygıtı bulunamadı, CPU devralıyor.")

scene.cycles.device = 'GPU' if cuda_found else 'CPU'
scene.cycles.samples = 32
scene.cycles.use_denoising = True

# Kamera Kurulumu
cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -16, 4)
cam_obj.rotation_euler = (math.radians(78), 0, 0)

# Işık
light_data = bpy.data.lights.new(name="SunLight", type='POINT')
light_data.energy = 8000
light_obj = bpy.data.objects.new("SunLight", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (-4, -8, 10)

# Şeffaf Güneş Çanağı (Yarım Küre)
bpy.ops.mesh.primitive_uv_sphere_add(radius=4.5, location=(0, 0, 0))
bowl = bpy.context.active_object
bowl.name = "Sun_Bowl"

# Basit küp kesici ile yarım çanak
bpy.ops.mesh.primitive_cube_add(size=10, location=(0, 0, 5))
cutter = bpy.context.active_object
cutter.name = "Cutter"

mod = bowl.modifiers.new(name="Cut", type='BOOLEAN')
mod.object = cutter
mod.operation = 'DIFFERENCE'
bpy.context.view_layer.objects.active = bowl
bpy.ops.object.modifier_apply(modifier="Cut")
bpy.data.objects.remove(cutter, do_unlink=True)

# Güneş Materyali
mat_sun = bpy.data.materials.new("SunGlass")
mat_sun.use_nodes = True
nodes = mat_sun.node_tree.nodes
nodes.clear()
glass = nodes.new("ShaderNodeBsdfGlass")
glass.inputs['Roughness'].default_value = 0.08
glass.inputs['IOR'].default_value = 1.15
glass.inputs['Color'].default_value = (1.0, 0.45, 0.05, 1.0)
emit = nodes.new("ShaderNodeEmission")
emit.inputs['Color'].default_value = (1.0, 0.25, 0.0, 1.0)
emit.inputs['Strength'].default_value = 1.2
add_node = nodes.new("ShaderNodeAddShader")
mat_sun.node_tree.links.new(glass.outputs['BSDF'], add_node.inputs[0])
mat_sun.node_tree.links.new(emit.outputs['Emission'], add_node.inputs[1])
out_node = nodes.new("ShaderNodeOutputMaterial")
mat_sun.node_tree.links.new(add_node.outputs['Shader'], out_node.inputs['Surface'])
bowl.data.materials.append(mat_sun)

# Pasif Fizik Alanı
bpy.ops.rigidbody.object_add()
bowl.rigid_body.type = 'PASSIVE'
bowl.rigid_body.collision_shape = 'MESH'

# Dünya Materyali
mat_earth = bpy.data.materials.new("EarthMat")
mat_earth.use_nodes = True
bsdf = mat_earth.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0.05, 0.35, 0.85, 1.0)
bsdf.inputs['Roughness'].default_value = 0.25

# 30 Adet Fiziksel Küre Dökülmesi
random.seed(42)
for i in range(30):
    rx = random.uniform(-1.8, 1.8)
    ry = random.uniform(-1.8, 1.8)
    rz = random.uniform(6.0, 14.0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.65, location=(rx, ry, rz))
    ball = bpy.context.active_object
    ball.data.materials.append(mat_earth)
    bpy.ops.rigidbody.object_add()
    ball.rigid_body.type = 'ACTIVE'
    ball.rigid_body.collision_shape = 'SPHERE'
    ball.rigid_body.mass = 2.0
    ball.rigid_body.restitution = 0.3

# Çıktı Dosyası
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.filepath = '/tmp/render_out.mp4'

print("Animasyon Render Ediliyor...")
bpy.ops.render.render(animation=True)
print("Render tamamlandi, dosya yazildi.")
"""

    with open("/tmp/scene.py", "w") as f:
        f.write(blender_script)

    # Blender çıktısını loglara canlı akıt
    process = subprocess.Popen(
        ["blender", "-b", "-P", "/tmp/scene.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Blender process failed with code {process.returncode}")

    # Dosyayı bul ve döndür
    all_files = os.listdir("/tmp")
    print(f"/tmp dizinindeki dosyalar: {all_files}")

    mp4_files = [f for f in all_files if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError(f"Video bulunamadi! /tmp icerigi: {all_files}")

    target = os.path.join("/tmp", mp4_files[0])
    with open(target, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("Modal GPU uzerinde Cycles render tetikleniyor...")
    video_bytes = render_scene.remote()
    with open("final_gpu_render.mp4", "wb") as f:
        f.write(video_bytes)
    print("Basarili! final_gpu_render.mp4 olusturuldu.")
