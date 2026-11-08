import subprocess
import modal

app = modal.App("shorts-blender-engine")

BLENDER_URL = "https://download.blender.org/release/Blender3.6/blender-3.6.18-linux-x64.tar.xz"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "xz-utils", "ffmpeg",
        # Blender'in headless modda bile link ettigi kutuphaneler
        "libgl1", "libegl1", "libxrender1", "libxi6", "libxxf86vm1",
        "libxfixes3", "libxext6", "libx11-6", "libxkbcommon0",
        "libsm6", "libice6",
    )
    .run_commands(
        f"wget -q {BLENDER_URL} -O /tmp/blender.tar.xz",
        "tar -xf /tmp/blender.tar.xz -C /opt",
        "ln -sf /opt/blender-3.6.18-linux-x64/blender /usr/local/bin/blender",
        "rm /tmp/blender.tar.xz",
    )
)

# ---------------------------------------------------------------------------
# Blender icinde calisacak sahne scripti (bpy)
# ---------------------------------------------------------------------------
BLENDER_SCRIPT = r'''
import bpy, bmesh, math, random, os

random.seed(7)

W, H = 1080, 1920
FPS = 30
SECONDS = 9
N_BALLS = 40
BALL_R = 0.35
BOWL_R = 2.0
OUT = "/tmp/frames/f_"

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = FPS * SECONDS
scene.render.fps = FPS

# ---------------- Dunya ----------------
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.005, 0.006, 0.02, 1)
bg.inputs[1].default_value = 1.0

# ---------------- Cam kase (ustu kesik, kalin cidarli) ----------------
bpy.ops.mesh.primitive_uv_sphere_add(radius=BOWL_R, segments=48, ring_count=32, location=(0, 0, 0))
bowl = bpy.context.active_object
bowl.name = "Bowl"
bm = bmesh.new()
bm.from_mesh(bowl.data)
cut = [v for v in bm.verts if v.co.z > 0.6]
bmesh.ops.delete(bm, geom=cut, context="VERTS")
bm.to_mesh(bowl.data)
bm.free()
for p in bowl.data.polygons:
    p.use_smooth = True

mod = bowl.modifiers.new("Solidify", "SOLIDIFY")
mod.thickness = 0.12
mod.offset = -1
bpy.context.view_layer.objects.active = bowl
bowl.select_set(True)
bpy.ops.object.modifier_apply(modifier="Solidify")

mat = bpy.data.materials.new("HotGlass")
mat.use_nodes = True
b = mat.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (1.0, 0.35, 0.05, 1)
b.inputs["Roughness"].default_value = 0.05
b.inputs["IOR"].default_value = 1.45
b.inputs["Transmission"].default_value = 1.0
b.inputs["Emission"].default_value = (1.0, 0.4, 0.05, 1)
b.inputs["Emission Strength"].default_value = 2.5
bowl.data.materials.append(mat)
bowl.visible_shadow = False

bpy.ops.rigidbody.object_add()
bowl.rigid_body.type = "PASSIVE"
bowl.rigid_body.collision_shape = "MESH"
bowl.rigid_body.friction = 0.5
bowl.rigid_body.restitution = 0.3
bowl.rigid_body.collision_margin = 0.001

# ---------------- Kureler ----------------
ball_mat = bpy.data.materials.new("Planet")
ball_mat.use_nodes = True
nt = ball_mat.node_tree
bsdf = nt.nodes["Principled BSDF"]
bsdf.inputs["Roughness"].default_value = 0.35
noise = nt.nodes.new("ShaderNodeTexNoise")
noise.inputs["Scale"].default_value = 3.0
noise.inputs["Detail"].default_value = 8.0
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.0
ramp.color_ramp.elements[0].color = (0.02, 0.08, 0.45, 1)
e1 = ramp.color_ramp.elements.new(0.48)
e1.color = (0.05, 0.25, 0.7, 1)
e2 = ramp.color_ramp.elements.new(0.55)
e2.color = (0.1, 0.45, 0.12, 1)
ramp.color_ramp.elements[-1].position = 1.0
ramp.color_ramp.elements[-1].color = (0.6, 0.5, 0.3, 1)
nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_R, segments=32, ring_count=16, location=(0, 0, 100))
proto = bpy.context.active_object
ball_mesh = proto.data
for p in ball_mesh.polygons:
    p.use_smooth = True
ball_mesh.materials.append(ball_mat)
bpy.data.objects.remove(proto)

coll = scene.collection
for i in range(N_BALLS):
    ang = random.uniform(0, math.tau)
    rad = random.uniform(0, 0.7)
    obj = bpy.data.objects.new("Ball%d" % i, ball_mesh)
    obj.location = (math.cos(ang) * rad, math.sin(ang) * rad, 3.5 + i * 0.85)
    obj.rotation_euler = (random.uniform(0, 3), random.uniform(0, 3), random.uniform(0, 3))
    coll.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = "ACTIVE"
    obj.rigid_body.collision_shape = "SPHERE"
    obj.rigid_body.mass = 1.0
    obj.rigid_body.friction = 0.4
    obj.rigid_body.restitution = 0.35
    obj.select_set(False)

rbw = scene.rigidbody_world
rbw.substeps_per_frame = 20
rbw.solver_iterations = 20
rbw.time_scale = 0.8
rbw.point_cache.frame_start = 1
rbw.point_cache.frame_end = FPS * SECONDS

# ---------------- Kamera (dikey 9:16) ----------------
cam_data = bpy.data.cameras.new("Cam")
cam_data.lens = 35
cam = bpy.data.objects.new("Cam", cam_data)
coll.objects.link(cam)
cam.location = (0, -14, 3)
target = bpy.data.objects.new("Target", None)
target.location = (0, 0, 3)
coll.objects.link(target)
tc = cam.constraints.new("TRACK_TO")
tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"
tc.up_axis = "UP_Y"
scene.camera = cam
# yavas yaklasma
cam.keyframe_insert("location", frame=1)
cam.location = (0, -12, 3)
cam.keyframe_insert("location", frame=FPS * SECONDS)

# ---------------- Isiklar ----------------
ld = bpy.data.lights.new("Key", "AREA")
ld.energy = 3000
ld.size = 8
key = bpy.data.objects.new("Key", ld)
key.location = (0, -8, 8)
key.rotation_euler = (math.radians(55), 0, 0)
coll.objects.link(key)

rd = bpy.data.lights.new("Rim", "POINT")
rd.energy = 2000
rim = bpy.data.objects.new("Rim", rd)
rim.location = (4, 3, 5)
coll.objects.link(rim)

# ---------------- Render ayarlari ----------------
scene.render.engine = "CYCLES"
scene.render.resolution_x = W
scene.render.resolution_y = H
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = OUT
scene.render.use_persistent_data = True

cy = scene.cycles
cy.samples = 32
cy.use_adaptive_sampling = True
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.max_bounces = 6
cy.diffuse_bounces = 2
cy.glossy_bounces = 4
cy.transmission_bounces = 6
cy.transparent_max_bounces = 4
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_indirect = 5.0

# GPU (CUDA) sec
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    gpu_found = False
    for d in prefs.devices:
        d.use = (d.type != "CPU")
        gpu_found = gpu_found or d.use
        print("DEVICE:", d.name, d.type, d.use)
    cy.device = "GPU" if gpu_found else "CPU"
except Exception as ex:
    print("GPU ayari basarisiz, CPU:", ex)
    cy.device = "CPU"
print("CYCLES DEVICE:", cy.device)

# Hafif parlama (bloom benzeri)
scene.use_nodes = True
tree = scene.node_tree
for n in list(tree.nodes):
    tree.nodes.remove(n)
rl = tree.nodes.new("CompositorNodeRLayers")
glare = tree.nodes.new("CompositorNodeGlare")
glare.glare_type = "FOG_GLOW"
glare.threshold = 1.0
glare.quality = "MEDIUM"
glare.size = 7
comp = tree.nodes.new("CompositorNodeComposite")
tree.links.new(rl.outputs["Image"], glare.inputs["Image"])
tree.links.new(glare.outputs["Image"], comp.inputs["Image"])

os.makedirs("/tmp/frames", exist_ok=True)
bpy.ops.render.render(animation=True)
print("RENDER BITTI")
'''


@app.function(gpu="T4", timeout=1500, image=image)
def render_scene() -> bytes:
    with open("/tmp/scene.py", "w") as f:
        f.write(BLENDER_SCRIPT)

    r = subprocess.run(
        ["blender", "-b", "-noaudio", "-P", "/tmp/scene.py"],
        capture_output=True, text=True,
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError("Blender hata verdi, loglara bak.")

    enc = subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30", "-i", "/tmp/frames/f_%04d.png",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-preset", "medium", "-movflags", "+faststart", "/tmp/out.mp4"],
        capture_output=True, text=True,
    )
    if enc.returncode != 0:
        print(enc.stderr[-3000:])
        raise RuntimeError("ffmpeg hata verdi.")

    with open("/tmp/out.mp4", "rb") as f:
        return f.read()


@app.local_entrypoint()
def main():
    data = render_scene.remote()
    with open("final_gpu_render.mp4", "wb") as f:
        f.write(data)
    print("Kaydedildi: final_gpu_render.mp4 (%d KB)" % (len(data) // 1024))
