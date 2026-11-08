import modal
import os

app = modal.App("shorts-blender-engine")

# Sanal ekran (xvfb) ve OpenGL sürücüleri eklenmiş imaj
image = (
    modal.Image.debian_slim()
    .apt_install(
        "wget", "xz-utils", "ffmpeg", "xvfb",
        "libxrender1", "libxi6", "libgl1", "libxkbcommon-x11-0",
        "libsm6", "libgl1-mesa-glx", "libgl1-mesa-dri"
    )
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
    import subprocess, glob

    blender_script = """
import bpy, math, random

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Çözünürlük & 5 saniyelik dökülme (150 kare)
scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 150

# EEVEE Motoru
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.use_gtao = True
scene.eevee.use_bloom = True
scene.eevee.use_ssr = True

# Kamera Kurulumu
cam_data = bpy.data.cameras.new("MainCam")
cam_data.lens = 45
cam_obj = bpy.data.objects.new("MainCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -26, 9)
cam_obj.rotation_euler = (math.radians(72), 0, 0)

# Işık
l1 = bpy.data.lights.new("Sun", 'SUN')
l1.energy = 4.5
o1 = bpy.data.objects.new("Sun", l1)
scene.collection.objects.link(o1)
o1.rotation_euler = (math.radians(50), math.radians(-25), 0)

# Çanak (Güneş Kabı)
bpy.ops.mesh.primitive_uv_sphere_add(radius=5.0, location=(0, 0, -2.5))
bowl = bpy.context.active_object
bowl.name = "SunBowl"

bpy.ops.mesh.primitive_cube_add(size=14, location=(0, 0, 4.0))
cutter = bpy.context.active_object
mod = bowl.modifiers.new("Cut", 'BOOLEAN')
mod.object = cutter
mod.operation = 'DIFFERENCE'
bpy.context.view_layer.objects.active = bowl
bpy.ops.object.modifier_apply(modifier="Cut")
bpy.data.objects.remove(cutter, do_unlink=True)

sol = bowl.modifiers.new("Thickness", 'SOLIDIFY')
sol.thickness = 0.35
bpy.ops.object.modifier_apply(modifier="Thickness")

mat_bowl = bpy.data.materials.new("GlassSun")
mat_bowl.use_nodes = True
mat_bowl.blend_method = 'BLEND'
bsdf_bowl = mat_bowl.node_tree.nodes.get("Principled BSDF")
bsdf_bowl.inputs['Base Color'].default_value = (1.0, 0.55, 0.1, 1.0)
bsdf_bowl.inputs['Transmission Weight'].default_value = 0.85
bsdf_bowl.inputs['Roughness'].default_value = 0.08
bsdf_bowl.inputs['Emission Color'].default_value = (1.0, 0.3, 0.0, 1.0)
bsdf_bowl.inputs['Emission Strength'].default_value = 1.0
bowl.data.materials.append(mat_bowl)

bpy.ops.rigidbody.object_add()
bowl.rigid_body.type = 'PASSIVE'
bowl.rigid_body.collision_shape = 'MESH'

# Küreler (Dünyalar)
mat_ball = bpy.data.materials.new("EarthMat")
mat_ball.use_nodes = True
bsdf_ball = mat_ball.node_tree.nodes.get("Principled BSDF")
bsdf_ball.inputs['Base Color'].default_value = (0.05, 0.42, 0.95, 1.0)
bsdf_ball.inputs['Roughness'].default_value = 0.15

random.seed(42)
for i in range(40):
    rx = random.uniform(-1.5, 1.5)
    ry = random.uniform(-1.5, 1.5)
    rz = 2.5 + (i * 0.6)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.60, location=(rx, ry, rz))
    ball = bpy.context.active_object
    ball.data.materials.append(mat_ball)
    bpy.ops.rigidbody.object_add()
    ball.rigid_body.type = 'ACTIVE'
    ball.rigid_body.collision_shape = 'SPHERE'
    ball.rigid_body.mass = 3.0
    ball.rigid_body.restitution = 0.25

# Çıktı formatı
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.filepath = '/tmp/render_'
scene.render.use_file_extension = True

print("Render basliyor...")
bpy.ops.render.render(animation=True)
print("Render tamamlandi.")
"""

    with open("/tmp/scene.py", "w") as f:
        f.write(blender_script)

    # xvfb ile sanal ekran kurarak EEVEE'nin OpenGL hatası vermesini engelliyoruz
    cmd = ["xvfb-run", "-a", "blender", "-b", "-P", "/tmp/scene.py"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)

    if res.returncode != 0:
        raise RuntimeError(f"Blender patladi, cikis kodu: {res.returncode}")

    # Üretilen MP4 dosyasını yakala
    files = glob.glob("/tmp/render_*.mp4")
    if not files:
        raise FileNotFoundError(f"Video bulunamadi! /tmp dizini: {os.listdir('/tmp')}")

    with open(files[0], "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    video_bytes = render_scene.remote()
    with open("final_gpu_render.mp4", "wb") as f:
        f.write(video_bytes)
    print("Islem basarili: final_gpu_render.mp4 olusturuldu.")
