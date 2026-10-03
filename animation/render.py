#!/usr/bin/env python3
"""
ONE SCHOOL, ONE DAY - 3D animation renderer
Renders the documentary plan as a detailed 3D animated film using a custom
software 3D engine (Python) + ffmpeg for encoding. All material is generated
from scratch -> 100% copyright free.

Scenes follow the shot list in:
 "One School, One Day - Let Yet Kone Documentary Plan (2).md"
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os, math, random, subprocess, sys

W, H, FPS = 1920, 1080, 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "scenes")
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

rng = random.Random(16092022)
np.random.seed(16092022)

# ----------------------------------------------------------------------------
# Tiny 3D engine: vertices, faces, perspective camera, painter's algorithm
# ----------------------------------------------------------------------------
def rot_x(v, a):
    c, s = math.cos(a), math.sin(a)
    y, z = v[..., 1] * c - v[..., 2] * s, v[..., 1] * s + v[..., 2] * c
    out = v.copy(); out[..., 1] = y; out[..., 2] = z
    return out

def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    x, z = v[..., 0] * c + v[..., 2] * s, -v[..., 0] * s + v[..., 2] * c
    out = v.copy(); out[..., 0] = x; out[..., 2] = z
    return out

def persp(project, focal=900.0):
    """project: Nx3 camera-space points -> (Nx2 screen, N depth)"""
    z = np.maximum(project[:, 2], 1e-4)
    sx = W / 2 + project[:, 0] * focal / z
    sy = H / 2 - project[:, 1] * focal / z
    return np.stack([sx, sy], axis=1), z

class Scene3D:
    def __init__(self, cam_pos, yaw, pitch, fov_focal=900.0):
        self.cam = np.array(cam_pos, dtype=float)
        self.yaw, self.pitch = yaw, pitch
        self.focal = fov_focal
        self.polys = []   # (verts_world Nx3, rgb, sort_depth_override or None)

    def add_poly(self, verts, rgb, depth=None):
        self.polys.append((np.asarray(verts, float), rgb, depth))

    def render(self, bg_top=(10, 12, 20), bg_bot=(30, 26, 40), sun_dir=None,
               ambient=0.35, draw_bg="sky"):
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        # background gradient
        if draw_bg == "sky":
            for yy in range(0, H, 4):
                t = yy / H
                col = tuple(int(bg_top[i] * (1 - t) + bg_bot[i] * t) for i in range(3))
                d.rectangle([0, yy, W, yy + 4], fill=col)
        elif draw_bg == "black":
            d.rectangle([0, 0, W, H], fill=(3, 3, 6))
        # transform polys to camera space
        items = []
        Rx_p = np.array([[math.cos(self.pitch), -math.sin(self.pitch), 0],
                         [math.sin(self.pitch),  math.cos(self.pitch), 0],
                         [0, 0, 1]])
        Ry_w = np.array([[math.cos(self.yaw), 0, math.sin(self.yaw)],
                         [0, 1, 0],
                         [-math.sin(self.yaw), 0, math.cos(self.yaw)]])
        Rt = Rx_p @ Ry_w
        for poly in self.polys:
            if len(poly) == 3:
                verts, rgb, ov = poly
            else:
                (verts, rgb), ov = poly, None
            vc = (verts - self.cam) @ Rt.T
            # back-face & frustum reject by depth
            dep = vc[:, 2].mean() if ov is None else ov
            if dep < 0.5:
                continue
            pts, _ = persp(vc, self.focal)
            # lighting
            if sun_dir is not None and len(verts) >= 3:
                n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
                ln = np.linalg.norm(n)
                if ln > 1e-9:
                    n = n / ln
                    ndl = max(0.0, float(np.dot(n, sun_dir)))
                    light = ambient + (1 - ambient) * ndl
                else:
                    light = 1.0
            else:
                light = 1.0
            col = tuple(min(255, int(c * light)) for c in rgb)
            items.append((dep, pts, col))
        items.sort(key=lambda it: -it[0])
        for dep, pts, col in items:
            d.polygon([tuple(p) for p in pts], fill=col)
        return img

# ----------------------------------------------------------------------------
# Primitive builders (all return list of (verts, rgb))
# ----------------------------------------------------------------------------
def box(cx, cy, cz, sx, sy, sz, rgb, rot=0.0, shade=True):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]], float)
    if rot:
        v = rot_y(v, rot)
    v = v + np.array([cx, cy, cz])
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    out = []
    shades = [1.0, 0.85, 0.75, 0.9, 0.8, 0.7] if shade else [1] * 6
    for f, s in zip(faces, shades):
        out.append((v[list(f)], tuple(int(c * s) for c in rgb)))
    return out

def pyramid(cx, cy, cz, sx, sz, h, rgb):
    apex = np.array([cx, cy + h, cz])
    b = np.array([[cx - sx/2, cy, cz - sz/2], [cx + sx/2, cy, cz - sz/2],
                  [cx + sx/2, cy, cz + sz/2], [cx - sx/2, cy, cz + sz/2]])
    out = []
    for i in range(4):
        out.append((np.stack([b[i], b[(i + 1) % 4], apex]),
                    tuple(int(c * (0.8 + 0.2 * (i % 2))) for c in rgb)))
    out.append((b[::-1], tuple(int(c * 0.5) for c in rgb)))
    return out

def cone_strip(cx, base_y, cz, r, h, rgb, segs=14, tilt=0.0, tilt_axis='x'):
    """thin cone approximating a rotor blade set / roof"""
    out = []
    top = np.array([cx, base_y + h, cz])
    ring = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        ring.append(np.array([cx + r * math.cos(a), base_y, cz + r * math.sin(a)]))
    ring = np.array(ring)
    if tilt:
        for i in range(len(ring)):
            pass
    for i in range(segs):
        out.append((np.stack([ring[i], ring[(i + 1) % segs], top]),
                    tuple(int(c * (0.7 + 0.3 * abs(math.sin(i)))) for c in rgb)))
    return out

def cylinder(cx, cy, cz, r, h, rgb, segs=10):
    out = []
    bot = []
    top = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        bot.append([cx + r * math.cos(a), cy, cz + r * math.sin(a)])
        top.append([cx + r * math.cos(a), cy + h, cz + r * math.sin(a)])
    for i in range(segs):
        j = (i + 1) % segs
        s = 0.6 + 0.4 * abs(math.cos(2 * math.pi * i / segs))
        out.append((np.array([bot[i], bot[j], top[j], top[i]]),
                    tuple(int(c * s) for c in rgb)))
    out.append((np.array(top), tuple(int(c * 1.0) for c in rgb)))
    return out

def ground_quad(size, rgb):
    return [(np.array([[-size, 0, -size], [size, 0, -size], [size, 0, size], [-size, 0, size]]), rgb)]

def polyline_road(x0, x1, y, zc, width, rgb):
    return [(np.array([[x0, y, zc - width], [x1, y, zc - width],
                        [x1, y, zc + width], [x0, y, zc + width]]), rgb)]

def tree(cx, cz, scale=1.0, kind=0):
    prims = []
    trunk_h = 1.6 * scale
    prims += box(cx, trunk_h / 2, cz, 0.25 * scale, trunk_h, 0.25 * scale, (92, 66, 42))
    if kind == 0:  # round-ish canopy from stacked boxes
        for i, (dy, s, sh) in enumerate([(1.4, 1.1, (46, 102, 52)), (1.9, 0.85, (58, 122, 62)), (2.3, 0.55, (70, 140, 72))]):
            prims += box(cx, dy * scale, cz, s * scale, s * 0.7 * scale, s * scale, sh)
    else:  # palm-like
        prims += box(cx, trunk_h / 2 + 0.6 * scale, cz, 0.16 * scale, 1.2 * scale, 0.16 * scale, (110, 82, 52))
        for k in range(6):
            a = k * math.pi / 3
            tip = np.array([cx + 1.3 * scale * math.cos(a), trunk_h + 1.5 * scale, cz + 1.3 * scale * math.sin(a)])
            base = np.array([cx, trunk_h + 1.2 * scale, cz])
            mid = (base + tip) / 2 + np.array([0, 0.25 * scale, 0])
            prims.append((np.stack([base + [0.12, 0, 0.12], base - [0.12, 0, 0.12], tip, mid]), (52, 118, 58)))
    return prims

def stupa(cx, cz, scale=1.0, rgb=(214, 178, 92)):
    """Myanmar-style pagoda: tiered terraces + octagonal spire"""
    prims = []
    y = 0
    tiers = [(6.0, 1.0), (4.6, 0.9), (3.4, 0.8), (2.4, 0.7)]
    for w, hh in tiers:
        prims += box(cx, y + hh / 2, cz, w * scale, hh, w * scale,
                     tuple(int(c * (0.85 + 0.05 * i)) for i, c in enumerate(rgb)))
        y += hh
    prims += cylinder(cx, y, cz, 1.1 * scale, 1.6 * scale, rgb, segs=8)
    y += 1.6 * scale
    prims += pyramid(cx, y, cz, 1.6 * scale, 1.6 * scale, 2.6 * scale, tuple(int(c * 0.95) for c in rgb))
    return prims

def monastery_compound(cx=0, cz=-14):
    """main teaching building + kutis + fence gate + courtyard"""
    prims = []
    # main building (long classroom block, tin roof)
    prims += box(cx - 6, 1.5, cz, 9, 3, 5, (150, 128, 96))
    prims += box(cx - 6, 3.35, cz, 9.6, 0.35, 5.6, (96, 96, 104))          # roof slab
    for i in range(4):                                                      # windows
        prims += box(cx - 9.2 + i * 2.2, 1.8, cz + 2.55, 1.1, 1.2, 0.06, (40, 60, 74))
    prims += box(cx - 6, 1.0, cz + 2.55, 1.2, 2.0, 0.1, (88, 60, 38))      # door
    # second building
    prims += box(cx + 6.5, 1.2, cz - 1, 5, 2.4, 4, (162, 140, 108))
    prims += pyramid(cx + 6.5, 2.4, cz - 1, 5.8, 4.8, 1.4, (122, 74, 52))
    # small kutis
    for (ox, oz) in [(cx + 11, cz + 3), (cx + 11, cz - 3)]:
        prims += box(ox, 0.9, oz, 2.2, 1.8, 2.2, (140, 118, 92))
        prims += pyramid(ox, 1.8, oz, 2.6, 2.6, 0.9, (110, 66, 46))
    # stupa
    prims += stupa(cx + 2.5, cz + 6, 0.8)
    # fence posts + gate
    for i in range(14):
        px = cx - 13 + i * 2
        prims += box(px, 0.5, cz + 9.5, 0.15, 1.0, 0.15, (120, 96, 70))
    prims += box(cx - 13, 0.5, cz + 9.5, 0.15, 1.0, 0.15, (120, 96, 70))
    # flag pole
    prims += box(cx - 10, 1.75, cz + 8, 0.08, 3.5, 0.08, (170, 170, 175))
    prims.append((np.array([[cx - 10, 3.5, cz + 8], [cx - 8.6, 3.3, cz + 8], [cx - 8.6, 2.7, cz + 8], [cx - 10, 2.9, cz + 8]]), (210, 60, 60)))
    return prims

def school_desk(cx, cy, cz, rgb=(176, 130, 78), rot=0.0):
    prims = []
    prims += box(cx, cy + 0.75, cz, 1.1, 0.07, 0.6, rgb, rot)
    prims += box(cx, cy + 0.45, cz - 0.22, 0.06, 0.55, 0.06, (120, 88, 52), rot)
    prims += box(cx, cy + 0.45, cz + 0.22, 0.06, 0.55, 0.06, (120, 88, 52), rot)
    # chair
    chx, chz = cx, cz + 0.75
    prims += box(chx, cy + 0.42, chz, 0.45, 0.05, 0.45, (150, 110, 66), rot)
    prims += box(chx, cy + 0.72, chz + 0.2, 0.45, 0.55, 0.05, (150, 110, 66), rot)
    for ox, oz in [(-0.18, -0.18), (0.18, -0.18), (-0.18, 0.18), (0.18, 0.18)]:
        prims += box(chx + ox, cy + 0.2, chz + oz, 0.05, 0.4, 0.05, (110, 80, 48), rot)
    return prims

def chalkboard(cy=0):
    prims = []
    prims += box(0, 1.5 + cy, -3.0, 3.4, 1.6, 0.08, (24, 66, 48))
    prims += box(0, 1.5 + cy, -2.96, 3.0, 1.2, 0.02, (34, 84, 60))
    # chalk lines
    for i, lw in enumerate([2.2, 1.6, 2.6]):
        prims += box(-0.5 + i * 0.1, 1.9 - i * 0.35 + cy, -2.93, lw, 0.05, 0.02, (230, 232, 228))
    return prims

def child_figure(cx, cz, shirt, skin=(222, 178, 140), scale=1.0, walk_phase=0.0, facing=0.0):
    """simple low-poly kid: head, torso, arms, legs"""
    prims = []
    s = scale
    leg_sw = math.sin(walk_phase) * 0.22 * s
    prims += box(cx - 0.12 * s, 0.32 * s, cz + leg_sw, 0.14 * s, 0.62 * s, 0.14 * s, (60, 70, 110))
    prims += box(cx + 0.12 * s, 0.32 * s, cz - leg_sw, 0.14 * s, 0.62 * s, 0.14 * s, (52, 62, 100))
    prims += box(cx, 0.95 * s, cz, 0.42 * s, 0.62 * s, 0.24 * s, shirt)
    arm_sw = math.sin(walk_phase + math.pi) * 0.18 * s
    prims += box(cx - 0.3 * s, 1.0 * s, cz + arm_sw, 0.1 * s, 0.5 * s, 0.1 * s, shirt)
    prims += box(cx + 0.3 * s, 1.0 * s, cz - arm_sw, 0.1 * s, 0.5 * s, 0.1 * s, shirt)
    prims += box(cx, 1.48 * s, cz, 0.3 * s, 0.32 * s, 0.28 * s, skin)
    prims += box(cx, 1.66 * s, cz - 0.02 * s, 0.32 * s, 0.12 * s, 0.3 * s, (30, 26, 24))  # hair
    return prims

def sandal_pair(cx, cz, rot=0.0, rgb=(150, 90, 50)):
    prims = []
    for off in (-0.14, 0.16):
        v = np.array([[-0.11, 0, -0.24], [0.11, 0, -0.24], [0.11, 0.05, 0.24], [-0.11, 0.05, 0.24]])
        v = rot_y(v, rot) + np.array([cx + off, 0.025, cz + off * 0.3])
        prims.append((v, rgb))
        strap = np.array([[-0.09, 0.05, -0.05], [0.09, 0.05, -0.05], [0.09, 0.07, 0.05], [-0.09, 0.07, 0.05]])
        prims.append((rot_y(strap, rot) + np.array([cx + off, 0.0, cz + off * 0.3]), tuple(int(c * 0.6) for c in rgb)))
    return prims

def helicopter_model(cx, cy, cz, yaw=0.0, bob=0.0, rotor_phase=0.0, damaged=False):
    """low-poly Mi-8 style helicopter"""
    prims = []
    body_rgb = (74, 84, 72) if not damaged else (60, 58, 54)
    parts = []
    parts += box(0, 0, 0, 1.1, 1.0, 3.2, body_rgb, rot=yaw)             # fuselage
    parts += box(0, 0.55, -0.4, 0.9, 0.5, 1.6, (52, 60, 54), rot=yaw)   # top spine
    parts += box(0, 0.1, 2.1, 0.35, 0.35, 1.6, body_rgb, rot=yaw)       # tail boom
    parts += box(0, 0.55, 2.7, 0.08, 0.9, 0.5, (60, 68, 62), rot=yaw)   # tail fin
    parts += box(0, -0.7, -0.6, 1.6, 0.12, 0.12, (40, 40, 44), rot=yaw) # skid L
    parts += box(0, -0.7, 0.6, 1.6, 0.12, 0.12, (40, 40, 44), rot=yaw)  # skid R
    for tz in (-0.6, 0.6):
        parts += box(0, -0.4, tz, 0.1, 0.55, 0.1, (44, 44, 48), rot=yaw)
    parts += box(0.0, 0.05, -1.65, 0.8, 0.7, 0.15, (120, 160, 170), rot=yaw)  # cockpit glass
    # main rotor: 5 blades spinning
    for k in range(5):
        a = rotor_phase + k * 2 * math.pi / 5
        bx, bz = 1.9 * math.cos(a), 1.9 * math.sin(a)
        parts.append((np.array([[0, 0.95, 0], [bx, 0.93, bz], [bx * 1.02, 0.92, bz * 1.02], [0.05, 0.94, 0.05]]) +
                      np.array([0, 0, 0]), (30, 30, 34)))
        parts += box(bx / 2, 0.93, bz / 2, abs(bx) + 0.15, 0.05, abs(bz) + 0.15, (34, 34, 38), rot=0.0)
    # tail rotor
    for k in range(3):
        a = rotor_phase * 1.6 + k * 2 * math.pi / 3
        ty = 0.55 * math.cos(a); tz = 0.0
        parts.append((np.array([[0, 0.55, 2.7], [0.12, 0.55 + ty, 2.7 + 0.35], [0.12, 0.55 - ty, 2.7 - 0.35]]), (36, 36, 40)))
    # rotate all by yaw around origin then translate
    out = []
    for verts, rgb in parts:
        vr = rot_y(verts, yaw) + np.array([cx, cy + bob, cz])
        out.append((vr, rgb))
    if damaged:
        for i in range(6):
            ang = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0.3, 1.4)
            px, pz = cx + rr * math.cos(ang), cz + rr * math.sin(ang)
            py = cy + bob + rng.uniform(-0.8, 0.4)
            smoke_grey = (70 - i * 6, 68 - i * 6, 66 - i * 6)
            out += box(px, py, pz, 0.5 + i * 0.12, 0.5 + i * 0.12, 0.5 + i * 0.12, smoke_grey)
    return out

# ----------------------------------------------------------------------------
# Text overlay helpers (PIL)
# ----------------------------------------------------------------------------
_FONT_CACHE = {}
def font(sz, bold=True):
    key = (sz, bold)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(FONT_B if bold else FONT_R, sz)
    return _FONT_CACHE[key]

def text_card(lines, path, dur_s, bg=(3, 3, 6), sizes=None, colors=None, pauses=None):
    """render static card video via ffmpeg"""
    png = path.replace(".mp4", ".png")
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = 0
    total_h = 0
    layout = []
    for i, ln in enumerate(lines):
        sz = (sizes[i] if sizes else 54)
        f = font(sz, bold=(i == 0))
        bbox = d.textbbox((0, 0), ln, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        layout.append((ln, f, tw, th, (colors[i] if colors else (235, 235, 235))))
        total_h += th + 26
    y = (H - total_h) // 2
    for ln, f, tw, th, col in layout:
        d.text(((W - tw) // 2, y), ln, font=f, fill=col)
        y += th + 26
    img.save(png)
    frames = int(dur_s * FPS)
    fade_in = min(15, frames // 4)
    fade_out = min(20, frames // 4)
    vf = (f"fade=t=in:st=0:d={fade_in/FPS:.2f},fade=t=out:st={max(0,(frames-fade_out))/FPS:.2f}:d={fade_out/FPS:.2f}")
    run_ffmpeg(["-loop", "1", "-t", f"{dur_s}", "-i", png,
                "-vf", f"{vf},format=yuv420p", "-r", str(FPS),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18", path])

def run_ffmpeg(args):
    cmd = ["ffmpeg", "-hide_banner", "-y"] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG ERROR:", r.stderr[-1500:])
        raise SystemExit(1)

def render_scene(name, frame_fn, n_frames, workers=1):
    """frame_fn(i)->PIL image ; writes mp4 with ffmpeg raw pipe"""
    path = os.path.join(OUT_DIR, name)
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(n_frames):
        img = frame_fn(i)
        proc.stdin.write(img.tobytes())
        if i % max(1, n_frames // 10) == 0:
            print(f"  {name}: {i}/{n_frames}", flush=True)
    proc.stdin.close()
    proc.wait()
    return path

def overlay_drawtext(scene_path, out_path, texts):
    """texts: list of (text, start, dur, x_expr, y, size, color, alpha_expr?)"""
    filters = []
    for t in texts:
        txt, st, du, x, y, size, col = t[:7]
        esc = txt.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
        filters.append(f"drawtext=fontfile={FONT_B}:text='{esc}':fontcolor={col}:fontsize={size}:x={x}:y={y}:enable='between(t,{st},{st+du})':alpha='if(lt(t-{st},0.6),(t-{st})/0.6,if(lt({st}+{du}-t,0.6),({st}+{du}-t)/0.6,1))'")
    vf = ",".join(filters)
    run_ffmpeg(["-i", scene_path, "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "19", out_path])

# ----------------------------------------------------------------------------
# Sky elements (2D composited behind 3D) - stars, sun/moon glow, clouds
# ----------------------------------------------------------------------------
def draw_stars(d, t, n=140, horizon_y=H):
    random.seed(42)
    for _ in range(n):
        x = random.uniform(0, W); y = random.uniform(0, horizon_y * 0.75)
        br = random.uniform(0.3, 1.0)
        tw = 0.75 + 0.25 * math.sin(t * 2 + x * 0.05 + y * 0.03)
        c = int(255 * br * tw)
        d.ellipse([x, y, x + 2, y + 2], fill=(c, c, min(255, c + 10)))

def draw_glow(img, cx, cy, r, color, intensity=1.0):
    """soft radial glow via cached sprite"""
    key = (color, r // 8)
    spr = _GLOW_CACHE.get(key)
    if spr is None:
        size = max(64, r * 2)
        spr = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(spr)
        c = size / 2
        steps = 36
        for i in range(steps, 0, -1):
            rr = size / 2 * i / steps
            a = int(90 * (1 - i / steps) ** 2)
            d.ellipse([c - rr, c - rr, c + rr, c + rr], fill=(color[0], color[1], color[2], a))
        _GLOW_CACHE[key] = spr
    sc = (r * 2) / spr.width
    im = spr.resize((max(2, int(spr.width * sc)), max(2, int(spr.height * sc)))) if abs(sc - 1) > 0.05 else spr
    if intensity != 1.0:
        al = im.split()[3].point(lambda p: int(p * min(1.6, max(0.2, intensity))))
        im = im.copy(); im.putalpha(al)
    base = img.convert("RGBA")
    base.paste(im, (int(cx - im.width / 2), int(cy - im.height / 2)), im)
    img.paste(base.convert("RGB"), (0, 0))

_GLOW_CACHE = {}
def add_birds(d, t, count=5, scale=1.0, alpha=200):
    for k in range(count):
        bx = (W * 0.15 + k * 130 + t * 28 * (1 + 0.12 * k)) % (W + 200) - 100
        by = H * 0.18 + 60 * math.sin(t * 0.7 + k * 2.1) + k * 22
        flap = 7 + 5 * math.sin(t * 9 + k * 1.7)
        s = 12 * scale
        col = (40, 40, 50, alpha) if alpha < 255 else (40, 40, 50)
        try:
            d.line([(bx - s, by), (bx, by - flap), (bx + s, by)], fill=col, width=2)
        except Exception:
            pass

def haze_band(y0, y1, col_top, col_bot, alpha=70):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dd = ImageDraw.Draw(ov)
    for yy in range(y0, y1, 3):
        tt = (yy - y0) / max(1, y1 - y0)
        c = tuple(int(col_top[i] * (1 - tt) + col_bot[i] * tt) for i in range(3)) + (alpha,)
        dd.rectangle([0, yy, W, yy + 3], fill=c)
    return ov

# ============================================================================
# SCENE 1 : DAWN VILLAGE ROAD  (children walking to school)  ~ 45 s
# ============================================================================
def scene_dawn():
    N = 45 * FPS
    STATIC = []
    # ground, road, paddy fields
    STATIC += ground_quad(120, (74, 96, 58))
    STATIC += polyline_road(-120, 120, 0.01, 0, 2.2, (128, 106, 82))
    for z in (-14, 14):
        STATIC += [(np.array([[-120, 0.02, z - 6], [120, 0.02, z - 6], [120, 0.02, z + 6], [-120, 0.02, z + 6]]), (96, 132, 74))]
        for r in range(9):
            STATIC += polyline_road(-120, 120, 0.03, z - 5 + r * 1.2, 0.18, (78, 112, 60))
    # village houses along road
    random.seed(7)
    for k in range(9):
        hx = -60 + k * 15 + random.uniform(-2, 2)
        hz = -8 - random.uniform(1, 3) if k % 2 == 0 else 8 + random.uniform(1, 3)
        STATIC += box(hx, 1.0, hz, 4.2, 2.0, 3.2, (158, 132, 100))
        STATIC += pyramid(hx, 2.0, hz, 5.0, 4.0, 1.3, (104, 62, 44))
        STATIC += box(hx + 1.2, 0.8, hz + (1.65 if hz < 0 else -1.65), 0.9, 1.4, 0.08, (80, 56, 40))
        for ox in (-1.2, 0.2):
            STATIC += box(hx + ox, 1.3, hz + (1.65 if hz < 0 else -1.65), 0.7, 0.7, 0.06, (48, 62, 74))
        STATIC += tree(hx + 3.2, hz + 0.5, 1.0, kind=random.choice([0, 1]))
    # monastery + stupa on hill ahead
    STATIC += [(np.array([[-30, 0, -46], [30, 0, -46], [30, 7, -34], [-30, 7, -34]]), (86, 108, 66))]
    STATIC += monastery_compound(cx=-2, cz=-40)
    STATIC += stupa(18, -50, 1.4)
    STATIC += tree(-24, -30, 1.4, 0); STATIC += tree(26, -28, 1.2, 1)
    for tx in (-16, -8, 8, 16, 22):
        STATIC += tree(tx, -24 + (tx % 5), 1.1, tx % 2)
    # cows by roadside
    for (bx_, bz_) in [(6, 5.5), (8.5, 6.2)]:
        STATIC += box(bx_, 0.75, bz_, 1.7, 1.0, 0.8, (208, 200, 190))
        STATIC += box(bx_ + 1.0, 1.15, bz_, 0.5, 0.5, 0.5, (214, 206, 196))
        for lx in (-0.6, 0.6):
            STATIC += box(bx_ + lx, 0.2, bz_ + 0.3, 0.12, 0.45, 0.12, (190, 182, 172))
            STATIC += box(bx_ + lx, 0.2, bz_ - 0.3, 0.12, 0.45, 0.12, (190, 182, 172))

    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[-40 + 34 * prog, 2.2 - 0.5 * math.sin(prog * 3.1), 8.5],
                     yaw=math.radians(-8 + 6 * prog), pitch=math.radians(2.5), fov_focal=820)
        sc.polys += STATIC
        # children walking (illustration figures, animated)
        shirts = [(196, 74, 60), (60, 96, 176), (222, 186, 74), (90, 150, 96), (170, 96, 160)]
        for k in range(5):
            phase = t * 5.2 + k * 1.3
            px = -14 + 26 * min(1.0, prog * 1.15) + k * 1.7 * math.sin(k + 1)
            pz = -0.9 + (k % 3 - 1) * 0.9
            sc.polys += child_figure(px, pz, shirts[k % 5], walk_phase=phase, scale=0.95 + 0.05 * (k % 3))
            sc.polys += box(px + 0.32, 1.15, pz + 0.2, 0.3, 0.42, 0.16, (60, 60, 90))
        sun_dir = np.array([-0.45, 0.28, 0.5]); sun_dir /= np.linalg.norm(sun_dir)
        dawn_t = min(1.0, prog * 1.6)
        top = (int(16 + 70 * dawn_t), int(14 + 90 * dawn_t), int(40 + 120 * dawn_t))
        bot = (int(120 + 90 * dawn_t), int(70 + 80 * dawn_t), int(50 + 50 * dawn_t))
        img = sc.render(bg_top=top, bg_bot=bot, sun_dir=sun_dir, ambient=0.45)
        d = ImageDraw.Draw(img)
        sxp = W * 0.72; syp = H * 0.46 - 40 * dawn_t
        draw_glow(img, sxp, syp, 260, (255, 190, 120), 1.0 + dawn_t * 0.6)
        d = ImageDraw.Draw(img)
        d.ellipse([sxp - 34, syp - 34, sxp + 34, syp + 34], fill=(255, 224, 170))
        add_birds(d, t, 6)
        ov = haze_band(y0=int(H*0.42), y1=int(H*0.62), col_top=(255, 180, 130),
                      col_bot=(120, 100, 80), alpha=int(60 * (1 - dawn_t * 0.5)))
        img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"), (0, 0))
        random.seed(11)
        for _ in range(26):
            x = random.uniform(W * 0.5, W); y = random.uniform(H * 0.3, H * 0.8)
            d.ellipse([x, y, x + 3, y + 3], fill=(255, 230, 190))
        return img
    return render_scene("s1_dawn.mp4", frame, N)

# ============================================================================
# SCENE 2 : CLASSROOM INTERIOR (chalkboard, desks, kids seen from behind)
# ============================================================================
def scene_classroom():
    N = 40 * FPS
    STATIC = []
    STATIC += [(np.array([[-7, 0.0, -3.05], [7, 0.0, -3.05], [7, 0.0, 7], [-7, 0.0, 7]]), (148, 118, 84))]
    STATIC += box(0, 1.6, -3.1, 14, 3.4, 0.15, (196, 178, 140))
    STATIC += box(-7.05, 1.6, 2, 0.15, 3.4, 10, (186, 168, 130))
    STATIC += box(7.05, 1.6, 2, 0.15, 3.4, 10, (176, 158, 122))
    STATIC += box(0, 3.45, 2, 14, 0.12, 10, (150, 140, 120))
    for wz in (-1, 2, 5):
        STATIC += box(6.95, 1.9, wz, 0.06, 1.5, 1.6, (210, 226, 236))
        STATIC += box(6.95, 1.9, wz, 0.06, 1.5, 0.08, (120, 100, 70))
    STATIC += chalkboard()
    DESKS = []
    random.seed(3)
    for r in range(3):
        for c in range(3):
            dx = -3.4 + c * 3.4; dz = 0.4 + r * 1.9
            DESKS += school_desk(dx, 0, dz)
            DESKS += box(dx - 0.1, 0.8, dz - 0.05, 0.4, 0.03, 0.3, (240, 240, 236))
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[0, 1.9 - 0.25 * prog, 6.5 - 2.2 * prog],
                     yaw=0.0, pitch=math.radians(4 + 3 * prog), fov_focal=850)
        sc.polys += STATIC
        # teacher at board
        ph = 0.0
        sc.polys += child_figure(-1.6, -2.3, (90, 120, 150), scale=1.25, walk_phase=0)
        arm_up = np.array([[ -1.2, 1.6, -2.2], [-1.2, 2.2, -2.95], [-1.05, 2.2, -2.95], [-1.05, 1.6, -2.2]])
        sc.polys.append((arm_up, (90, 120, 150)))
        # chalk writing appears over time on the board (small white strips)
        n_w = int(prog * 9)
        for k in range(n_w):
            wx = -1.2 + (k % 3) * 1.1; wy = 2.05 - (k // 3) * 0.33
            sc.polys += box(wx, wy, -2.9, 0.85 + 0.15 * math.sin(k), 0.045, 0.02, (236, 238, 232))
        sc.polys += DESKS
        shirts = [(200, 90, 70), (70, 110, 190), (226, 190, 80), (100, 160, 100), (180, 100, 170), (90, 150, 160)]
        for r in range(3):
            for c in range(3):
                dx = -3.4 + c * 3.4; dz = 0.4 + r * 1.9
                kid_shirt = shirts[(r * 3 + c) % 6]
                if not (r == 2 and c == 2 and prog > 0.75):  # one empty chair later
                    sc.polys += child_figure(dx, dz + 0.75, kid_shirt, scale=0.92,
                                             walk_phase=(0.6 * math.sin(t * 1.3 + r + c)))
        sun_dir = np.array([0.75, -0.15, -0.4]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(70, 80, 95), bg_bot=(90, 90, 90), sun_dir=sun_dir, ambient=0.55, draw_bg="black")
        d = ImageDraw.Draw(img)
        # volumetric light shafts from windows
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(ov)
        for wz_sx in (1500, 1150, 850):
            sway = 18 * math.sin(t * 0.5 + wz_sx)
            dd.polygon([(wz_sx, 300), (wz_sx + 90, 320), (wz_sx - 320 + sway, 1080), (wz_sx - 460 + sway, 1080)],
                       fill=(255, 240, 200, 26))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(img)
        # floating chalk dust particles
        random.seed(5)
        for _ in range(45):
            x = random.uniform(W * 0.2, W * 0.8); y = random.uniform(H * 0.25, H * 0.75)
            drift = (t * 12 + x) % 60
            xx = x + 10 * math.sin(t * 0.8 + y)
            yy = y - drift * 0.4
            br = int(150 + 80 * math.sin(t + x + y))
            d.ellipse([xx, yy, xx + 2, yy + 2], fill=(br, br, min(255, br + 20)))
        return img
    return render_scene("s2_classroom.mp4", frame, N)

# ============================================================================
# SCENE 3 : WALL CLOCK -> 1:00 PM (3D clock, camera pushes in)
# ============================================================================
def scene_clock():
    N = 18 * FPS
    def frame(i):
        t = i / FPS
        prog = i / N
        zoom = 5.2 - 3.4 * prog
        sc = Scene3D(cam_pos=[0, 1.6, zoom], yaw=0, pitch=0, fov_focal=900)
        sc.polys += box(0, 1.6, -0.4, 8, 5, 0.2, (206, 190, 158))   # wall
        sc.polys += cylinder(0, 1.1, -0.25, 1.15, 0.18, (60, 46, 30), segs=24)  # rim lying? build upright below
        # upright clock face: ring of segments
        import numpy as _np
        segs = 36
        R = 1.15
        ring_bots, ring_tops = [], []
        for k in range(segs):
            a1 = 2 * math.pi * k / segs
            ring_bots.append([R * math.cos(a1), 1.6 + R * math.sin(a1), -0.18])
            ring_tops.append([(R + 0.12) * math.cos(a1), 1.6 + (R + 0.12) * math.sin(a1), -0.18])
        for k in range(segs):
            j = (k + 1) % segs
            sc.polys.append((_np.array([ring_bots[k], ring_bots[j], ring_tops[j], ring_tops[k]]), (82, 58, 34)))
        face = []
        for k in range(segs):
            a1 = 2 * math.pi * k / segs
            face.append([R * math.cos(a1), 1.6 + R * math.sin(a1), -0.1])
        sc.polys.append((_np.array(face), (238, 234, 222)))
        for k in range(segs):
            a1 = 2 * math.pi * k / segs
            l_ = 0.1 if k % 3 else 0.18
            x1, y1 = (R - l_) * math.cos(a1), 1.6 + (R - l_) * math.sin(a1)
            x2, y2 = R * 0.97 * math.cos(a1), 1.6 + R * 0.97 * math.sin(a1)
            sc.polys.append((_np.array([[x1 - 0.015, y1, -0.09], [x1 + 0.015, y1, -0.09], [x2 + 0.015, y2, -0.09], [x2 - 0.015, y2, -0.09]]), (40, 40, 44)))
        # hands: time runs 10:08 -> 13:00 across the scene
        mins = 10 * 60 + 8 + prog * (13 * 60 - (10 * 60 + 8))
        ha = math.pi / 2 - (mins / 720) * 2 * math.pi
        ma = math.pi / 2 - ((mins % 60) / 60) * 2 * math.pi
        def hand(angle, length, wdt, rgb):
            hx, hy = length * math.cos(angle), length * math.sin(angle)
            tx, ty = -wdt * math.sin(angle), wdt * math.cos(angle)
            pts = _np.array([[tx, 1.6 + ty, -0.05], [-tx, 1.6 - ty, -0.05],
                             [hx + tx, 1.6 + hy + ty, -0.05], [hx - tx, 1.6 + hy - ty, -0.05]])
            return [(pts, rgb)]
        sc.polys += hand(ha, 0.62, 0.05, (30, 30, 34))
        sc.polys += hand(ma, 0.95, 0.032, (30, 30, 34))
        sc.polys += cylinder(0, 1.55, -0.06, 0.06, 0.06, (200, 40, 40), segs=8)
        sun_dir = np.array([0.3, 0.5, 0.8]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(30, 26, 22), bg_bot=(46, 38, 30), sun_dir=sun_dir, ambient=0.5)
        return img
    return render_scene("s3_clock.mp4", frame, N)

# ============================================================================
# SCENE 4 : 3D MAP ANIMATION (Sagaing, Let Yet Kone, four helicopter markers)
# ============================================================================
def scene_map():
    N = 24 * FPS
    import numpy as _np
    STATIC = []
    # terrain tile grid (green/brown patches, slight elevation noise)
    random.seed(9)
    TS = 4
    for gx in range(-6, 7):
        for gz in range(-6, 7):
            hgt = 0.35 + 0.5 * abs(math.sin(gx * 0.8) * math.cos(gz * 0.6)) + random.uniform(0, 0.25)
            base = (88, 122, 70) if (gx + gz) % 3 else (112, 132, 74)
            if gx < -2 and gz > 2:
                base = (120, 104, 78)  # dry patch
            x0, z0 = gx * TS, gz * TS
            v = np.array([[x0, 0, z0], [x0 + TS, 0, z0], [x0 + TS, 0, z0 + TS], [x0, 0, z0 + TS],
                          [x0 + TS / 2, hgt * 2.2, z0 + TS / 2]], float)
            # 4 slope triangles + flat quad
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                sh = 0.75 + 0.25 * ((a + b) % 2)
                STATIC.append((v[[a, b, 4]], tuple(int(c * sh) for c in base)))
            STATIC.append((v[:4][::-1], tuple(int(c * 0.6) for c in base)))
    # river (irrawaddy stylised): blue ribbon quads along a curve
    for s in range(26):
        tt2 = s / 25
        rx = -24 + 48 * tt2
        rz = 18 - 34 * tt2 + 6 * math.sin(tt2 * 6)
        col = (70, 120, 160) if s % 2 else (80, 132, 172)
        STATIC.append((np.array([[rx - 2, 0.15, rz - 1.4], [rx + 2, 0.15, rz - 1.4],
                       [rx + 2, 0.15, rz + 1.4], [rx - 2, 0.15, rz + 1.4]]), col))
    # road line to Ye-U
    for s in range(18):
        tt2 = s / 17
        rx = -6 + 26 * tt2; rz = -2 - 16 * tt2
        STATIC.append((np.array([[rx - 1.2, 0.2, rz], [rx + 1.2, 0.2, rz - 0.9],
                       [rx + 1.2, 0.2, rz + 0.9]]), (150, 138, 110)))
    # village marker: monastery cluster at Let Yet Kone
    lx, lz = -6, -2
    STATIC += box(lx, 0.5, lz, 1.6, 1.0, 1.2, (172, 148, 112))
    STATIC += pyramid(lx, 1.0, lz, 2.0, 1.6, 0.7, (120, 70, 50))
    STATIC += stupa(lx + 2.2, lz + 1.2, 0.32)
    for ox, oz in [(-2.4, 1.8), (2.6, -1.6), (-1.8, -2.4), (3.2, 2.2)]:
        STATIC += box(lx + ox, 0.35, lz + oz, 0.9, 0.7, 0.8, (158, 138, 108))
        STATIC += pyramid(lx + ox, 0.7, lz + oz, 1.15, 1.0, 0.45, (112, 66, 46))
    # Ye-U town marker (destination)
    STATIC += box(20, 0.6, -18, 2.2, 1.2, 2.0, (150, 140, 130))
    STATIC += box(22, 0.4, -16, 1.4, 0.8, 1.4, (140, 130, 120))
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[-26 + 14 * prog, 26 - 8 * prog, 26 - 10 * prog],
                     yaw=math.radians(38), pitch=math.radians(-48 + 8 * prog), fov_focal=950)
        sc.polys += STATIC
        # pin above school (bobbing)
        pin_y = 3.2 + 0.4 * math.sin(t * 2.4)
        sc.polys += cylinder(lx, 1.6, lz, 0.07, 1.4, (210, 60, 50), segs=8)
        sc.polys += pyramid(lx, pin_y, lz, 0.7, 0.7, -0.9, (222, 62, 52))
        # four helicopter markers arriving from NE, two descend to land, two circle
        appear = prog * 4
        for k in range(4):
            tk = max(0.0, min(1.0, appear - k * 0.35))
            if tk <= 0: continue
            start = np.array([26 + k * 3, 7.5, 22 + k * 2])
            if k < 2:  # landing pair
                endp = np.array([lx + (-2 if k == 0 else 2.5), 0.4, lz + 6.5])
                cur = start + (endp - start) * min(1.0, tk * 1.2)
            else:      # circling pair
                ang = (tk - 1) * 3.4 + k * 2.2
                cx_ = lx + 10 * math.cos(ang) * (1 - 0.4 * tk)
                cz_ = lz + 10 * math.sin(ang) * (1 - 0.4 * tk)
                cur = np.array([cx_, 6.0 - 1.2 * math.sin(ang * 2), cz_])
            sc.polys += box(cur[0], cur[1], cur[2], 0.9, 0.35, 1.6, (46, 52, 48), rot=math.atan2(lx - cur[0], lz - cur[2]))
            sc.polys += box(cur[0], cur[1] + 0.3, cur[2], 2.6, 0.06, 0.25, (30, 30, 34), rot=t * 30 + k)
            # shadow
            sc.polys.append((np.array([[cur[0]-0.8, 0.12, cur[2]-0.8], [cur[0]+0.8, 0.12, cur[2]-0.8],
                                       [cur[0]+0.8, 0.12, cur[2]+0.8], [cur[0]-0.8, 0.12, cur[2]+0.8]]), (40, 50, 36)))
        sun_dir = np.array([0.4, 0.7, 0.3]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(24, 34, 52), bg_bot=(52, 62, 70), sun_dir=sun_dir, ambient=0.55)
        d = ImageDraw.Draw(img)
        # graticule overlay lines (map feel)
        for gxx in range(0, W, 160):
            d.line([(gxx + 10 * math.sin(t*0.3), 0), (gxx, H)], fill=(255, 255, 255, 8), width=1)
        return img
    return render_scene("s4_map.mp4", frame, N)

# ============================================================================
# SCENE 5 : HELICOPTERS OVER THE COMPOUND (no explosion SFX per plan; visual
#           flyover, dust wash, then fade toward silence)
# ============================================================================
def scene_helis():
    N = 30 * FPS
    smoke_parts = []
    random.seed(16)
    for _ in range(14):
        smoke_parts.append((random.uniform(-8, 8), random.uniform(0, 6), random.uniform(-18, -8),
                            random.uniform(0.6, 1.6), random.uniform(0, 6)))
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[16, 3.2 + 1.5 * math.sin(prog * 2), 12],
                     yaw=math.radians(-35 - 10 * prog), pitch=math.radians(6), fov_focal=820)
        sc.polys += ground_quad(90, (96, 112, 66))
        sc.polys += polyline_road(-60, 60, 0.01, 16, 2.0, (128, 106, 82))
        sc.polys += monastery_compound(cx=-2, cz=-14)
        sc.polys += stupa(20, -30, 1.2)
        random.seed(7)
        for k in range(6):
            hx = -30 + k * 12; hz = 22 + random.uniform(-2, 2)
            sc.polys += box(hx, 1.0, hz, 4.2, 2.0, 3.2, (158, 132, 100))
            sc.polys += pyramid(hx, 2.0, hz, 5.0, 4.0, 1.3, (104, 62, 44))
        for tx in (-24, -14, 10, 24, 30):
            sc.polys += tree(tx, 6, 1.2, tx % 2)
        # four helis: two descending to land, two orbiting
        phase = t * 26
        desc = min(1.0, prog * 1.8)
        for k in range(4):
            if k < 2:
                cx_ = -8 + k * 9
                cy_ = 16 - 13.2 * desc
                cz_ = -2 + k * 3
                yaw_ = math.radians(150)
            else:
                ang = phase * 0.045 + k * math.pi
                cx_ = -2 + 16 * math.cos(ang)
                cz_ = -12 + 12 * math.sin(ang)
                cy_ = 9.5 + 1.2 * math.sin(phase * 0.1 + k)
                yaw_ = ang + math.pi / 2
            sc.polys += helicopter_model(cx_, cy_, cz_, yaw=yaw_, bob=0.12 * math.sin(phase * 0.2 + k), rotor_phase=phase + k)
        # rotor-wash dust swirls near ground under landing helis
        d0 = None
        sun_dir = np.array([-0.5, 0.55, 0.4]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(92, 120, 150), bg_bot=(150, 160, 150), sun_dir=sun_dir, ambient=0.5)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(ov)
        for (sx_, sy_, sz_, sr_, seed_) in smoke_parts:
            adv = (prog * 3 + seed_ * 0.1) % 1.0
            px_ = sx_ + 6 * adv; py_ = sy_ + 5 * adv; pz_ = sz_ - 4 * adv
            vc = np.array([[px_, py_, pz_]])
            dep = vc[:, 2].mean()
            # project manually
            Rt = np.array([[math.cos(sc.pitch), -math.sin(sc.pitch), 0],
                           [math.sin(sc.pitch), math.cos(sc.pitch), 0], [0, 0, 1]]) @ \
                 np.array([[math.cos(sc.yaw), 0, math.sin(sc.yaw)], [0, 1, 0], [-math.sin(sc.yaw), 0, math.cos(sc.yaw)]])
            vcc = (vc - sc.cam) @ Rt.T
            zc = max(vcc[0, 2], 1e-3)
            X = W / 2 + vcc[0, 0] * sc.focal / zc; Y = H / 2 - vcc[0, 1] * sc.focal / zc
            rr = sr_ * sc.focal / zc * (0.6 + adv)
            a = int(90 * (1 - adv) * min(1.0, desc * 2))
            if zc > 0.5 and 0 < X < W and 0 < Y < H:
                dd.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=(190, 178, 150, a))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        # late-scene desaturation toward the "silence" beat
        if prog > 0.8:
            f = (prog - 0.8) / 0.2
            gray = Image.eval(img, lambda x: x)
            from PIL import ImageEnhance
            img = ImageEnhance.Color(img).enhance(1 - 0.5 * f)
        return img
    return render_scene("s5_helis.mp4", frame, N)
