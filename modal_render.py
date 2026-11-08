import modal
import os

app = modal.App("shorts-blender-engine")

# Önbelleklenmiş Blender ve grafik bağımlılıkları
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
    timeout=600,
    image=image
)
def render_scene():
    import subprocess, sys

    blender_script = """
import bpy, math, random

print("=== SAHNE HAZIRLANIYOR ===")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Dikey 9:16 Shorts (15 Saniye = 450 Kare)
scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 450

# Hızlı ve Net GPU Cycles Ayarları
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.compute_device_type = 'CUDA'
cprefs.get_devices()
for d in cprefs.devices:
    d.use = True
scene.cycles.device = 'GPU'
scene.cycles.samples = 32
scene.cycles.use_denoising = True

# 1. KAMERA: Geniş açı, tam kadraj (Çanak alt yarıda, dökülenler üstte)
cam_data = bpy.data.cameras.new("MainCam")
cam_data.lens = 38
cam_obj = bpy.data.objects.new("MainCam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -32, 11)
cam_obj.rotation_euler = (math.radians(72), 0, 0)

# 2. STÜDYO IŞIĞI
l1 = bpy.data.lights.new("KeyLight", 'AREA')
l1.energy = 9000
l1.size = 12.0
o1 = bpy.data.objects.new("KeyLight", l1)
scene.collection.objects.link(o1)
o1.location = (-10, -18, 22)
o1.rotation_euler = (math.radians(45), math.radians(-25), 0)

l2 = bpy.data.lights.new("RimLight", 'POINT')
l2.energy = 5000
o2 = bpy.data.objects.new("RimLight", l2)
scene.collection.objects.link(o2)
o2.location = (10, 12, 10)

# 3. GÜNEŞ ÇANAĞI (Şeffaf Akkor Cam)
bpy.ops.mesh.primitive_uv_sphere_add(radius=5.5, location=(0, 0, -3.5))
bowl = bpy.context.active_object
bowl.name = "SunBowl"

# Çanağın üstünü kesme
bpy.ops.mesh.primitive_cube_add(size=14, location=(0, 0, 4.0))
cutter = bpy.context.active_object
mod = bowl.modifiers.new("Cut", 'BOOLEAN')
mod.object = cutter
mod.operation = 'DIFFERENCE'
bpy.context.view_layer.objects.active = bowl
bpy.ops.object.modifier_apply(modifier="Cut")
bpy.data.objects.remove(cutter, do_unlink=True)

# Gerçek cam et kalınlığı
sol = bowl.modifiers.new("Thickness", 'SOLIDIFY')
sol.thickness = 0.35
bpy.ops.object.modifier_apply(modifier="Thickness")

# Akkor Şeffaf Cam Materyali
mat_bowl = bpy.data.materials.new("GlassSun")
mat_bowl.use_nodes = True
nodes = mat_bowl.node_tree.nodes
nodes.clear()

glass = nodes.new("ShaderNodeBsdfGlass")
glass.inputs['Roughness'].default_value = 0.04
glass.inputs['IOR'].default_value = 1.15
glass.inputs['Color'].default_value = (1.0, 0.60, 0.15, 1.0)

emit = nodes.new("ShaderNodeEmission")
emit.inputs['Color'].default_value = (1.0, 0.28, 0.0, 1.0)
emit.inputs['Strength'].default_value = 1.4

add_node = nodes.new("ShaderNodeAddShader")
mat_bowl.node_tree.links.new(glass.outputs['BSDF'], add_node.inputs[0])
mat_bowl.node_tree.links.new(emit.outputs['Emission'], add_node.inputs[1])

out_node = nodes.new("ShaderNodeOutputMaterial")
mat_bowl.node_tree.links.new(add_node.outputs['Shader'], out_node.inputs['Surface'])
bowl.data.materials.append(mat_bowl)

# Çanak Fizik Alanı
bpy.ops.rigidbody.object_add()
bowl.rigid_body.type = 'PASSIVE'
bowl.rigid_body.collision_shape = 'MESH'

# 4. DÜNYALAR: Parlak Mavi PBR Küreler
mat_ball = bpy.data.materials.new("EarthMat")
mat_ball.use_nodes = True
bsdf_ball = mat_ball.node_tree.nodes.get("Principled BSDF")
bsdf_ball.inputs['Base Color'].default_value = (0.05, 0.42, 0.95, 1.0)
bsdf_ball.inputs['Roughness'].default_value = 0.18
bsdf_ball.inputs['Metallic'].default_value = 0.05

# 70 Adet Kürenin Zamana Yaygın Olarak Dökülmesi
random.seed(42)
for i in range(70):
    rx = random.uniform(-1.8, 1.8)
    ry = random.uniform(-1.8, 1.8)
    # Z yüksekliği kademeli verilerek dökülme hissi oluşturulur
    rz = 3.0 + (i * 0.75)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.62, location=(rx, ry, rz))
    ball = bpy.context.active_object
    ball.data.materials.append(mat_ball)
    bpy.ops.rigidbody.object_add()
    ball.rigid_body.type = 'ACTIVE'
    ball.rigid_body.collision_shape = 'SPHERE'
    ball.rigid_body.mass = 3.5
    ball.rigid_body.restitution = 0.22

# 5. ÇIKTI YOLU
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.filepath = '/tmp/render_out.mp4'

print("Cycles render baslatildi...")
bpy.ops.render.render(animation=True)
print("Render basariyla bitti.")
"""

    with open("/tmp/scene.py", "w") as f:
        f.write(blender_script)

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

    mp4_files = [f for f in os.listdir("/tmp") if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError("Video bulunamadi!")

    target = os.path.join("/tmp", mp4_files[0])
    with open(target, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    video_bytes = render_scene.remote()
    with open("final_gpu_render.mp4", "wb") as f:
        f.write(video_bytes)
    print("Islem tamamlandi: final_gpu_render.mp4 kaydedildi.")
