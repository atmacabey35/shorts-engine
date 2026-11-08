import modal
import os

app = modal.App("shorts-blender-engine")

# Blender 3.6 LTS + CUDA + FFmpeg içeren Linux GPU imajı
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
    import subprocess

    blender_script = """
import bpy, math, random

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# Dikey Shorts Ayarları
scene.render.resolution_x = 1080
scene.render.resolution_y = 1920
scene.render.fps = 30
scene.frame_start = 1
scene.frame_end = 90  # 3 saniyelik hızlı test

# GPU Cycles Motoru
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.compute_device_type = 'CUDA'
cprefs.get_devices()
for d in cprefs.devices:
    d.use = True
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.cycles.use_denoising = True

# Kamera (Yukarıdan açılı sinematik)
cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -18, 5)
cam_obj.rotation_euler = (math.radians(78), 0, 0)

# Key Light
light_data = bpy.data.lights.new(name="SunLight", type='POINT')
light_data.energy = 9000
light_obj = bpy.data.objects.new("SunLight", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (-5, -10, 12)

# Güneş Şeffaf Kap
bpy.ops.mesh.primitive_uv_sphere_add(radius=5.0, location=(0, 0, 0))
bowl = bpy.context.active_object
bpy.ops.object.modifier_add(type='BOOLEAN')
bpy.ops.mesh.primitive_cube_add(size=12, location=(0, 0, 6))
cutter = bpy.context.active_object
cutter.display_type = 'WIRE'
bowl.modifiers["Boolean"].object = cutter
bowl.modifiers["Boolean"].operation = 'DIFFERENCE'

# Şeffaf Akkor Cam Materyali
mat_sun = bpy.data.materials.new("SunGlass")
mat_sun.use_nodes = True
nodes = mat_sun.node_tree.nodes
nodes.clear()
glass = nodes.new("ShaderNodeBsdfGlass")
glass.inputs['Roughness'].default_value = 0.08
glass.inputs['IOR'].default_value = 1.15
glass.inputs['Color'].default_value = (1.0, 0.45, 0.05, 1.0)
emit = nodes.new("ShaderNodeEmission")
emit.inputs['Color'].default_value = (1.0, 0.2, 0.0, 1.0)
emit.inputs['Strength'].default_value = 1.5
add_node = nodes.new("ShaderNodeAddShader")
mat_sun.node_tree.links.new(glass.outputs['BSDF'], add_node.inputs[0])
mat_sun.node_tree.links.new(emit.outputs['Emission'], add_node.inputs[1])
out_node = nodes.new("ShaderNodeOutputMaterial")
mat_sun.node_tree.links.new(add_node.outputs['Shader'], out_node.inputs['Surface'])
bowl.data.materials.append(mat_sun)

bpy.ops.rigidbody.object_add()
bowl.rigid_body.type = 'PASSIVE'
bowl.rigid_body.collision_shape = 'MESH'

# Dünya PBR Materyali
mat_earth = bpy.data.materials.new("EarthMat")
mat_earth.use_nodes = True
bsdf = mat_earth.node_tree.nodes.get("Principled BSDF")
bsdf.inputs['Base Color'].default_value = (0.05, 0.35, 0.85, 1.0)
bsdf.inputs['Roughness'].default_value = 0.20
bsdf.inputs['Metallic'].default_value = 0.10

# Yağan Dünyalar
random.seed(42)
for i in range(45):
    rx = random.uniform(-2.2, 2.2)
    ry = random.uniform(-2.2, 2.2)
    rz = random.uniform(7.0, 17.0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.75, location=(rx, ry, rz))
    ball = bpy.context.active_object
    ball.data.materials.append(mat_earth)
    bpy.ops.rigidbody.object_add()
    ball.rigid_body.type = 'ACTIVE'
    ball.rigid_body.collision_shape = 'SPHERE'
    ball.rigid_body.mass = 5.0
    ball.rigid_body.restitution = 0.32

# Çıktı formatı ve yolu
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.filepath = '/tmp/render_'
scene.render.use_file_extension = True

bpy.ops.render.render(animation=True)
"""
    with open("/tmp/scene.py", "w") as f:
        f.write(blender_script)

    subprocess.run(["blender", "-b", "-P", "/tmp/scene.py"], check=True)

    # Blender'ın ürettiği dosyayı bul
    files = [f for f in os.listdir("/tmp") if f.startswith("render_") and f.endswith(".mp4")]
    if not files:
        raise FileNotFoundError("Blender video üretemedi!")
    
    target_path = os.path.join("/tmp", files[0])
    with open(target_path, "rb") as f:
        return f.read()

@app.local_entrypoint()
def main():
    print("Modal GPU üzerinde Cycles render başlatılıyor...")
    video_bytes = render_scene.remote()
    with open("final_gpu_render.mp4", "wb") as f:
        f.write(video_bytes)
    print("Render tamamlandı -> final_gpu_render.mp4 kaydedildi.")
