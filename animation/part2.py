#!/usr/bin/env python3
"""Part 2: scenes 6-8, procedural audio, final assembly."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import math, random, os, wave, sys
import render as R
from render import (Scene3D, box, pyramid, cylinder, ground_quad, polyline_road,
                    tree, stupa, monastery_compound, school_desk, chalkboard,
                    child_figure, sandal_pair, draw_glow, add_birds,
                    render_scene, text_card, overlay_drawtext, run_ffmpeg,
                    W, H, FPS, OUT_DIR)

# ============================================================================
# SCENE 6 : EVENING - column leaving toward Ye-U with detainees (restrained)
# ============================================================================
def scene_evening():
    N = 30 * FPS
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[10 - 4 * prog, 2.6, 14], yaw=R.math.radians(-20),
                     pitch=R.math.radians(4), fov_focal=850)
        sc.polys += ground_quad(90, (96, 96, 60))
        sc.polys += polyline_road(-90, 90, 0.01, 0, 3.2, (122, 100, 78))
        sc.polys += monastery_compound(cx=-4, cz=-16)
        random.seed(7)
        for k in range(5):
            hx = -34 + k * 16
            sc.polys += box(hx, 1.0, 10, 4.2, 2.0, 3.2, (150, 124, 94))
            sc.polys += pyramid(hx, 2.0, 10, 5.0, 4.0, 1.3, (100, 60, 42))
        move = -30 + 46 * prog
        for ti, tx in enumerate([move, move + 9]):
            tz = -1.2 + ti * 2.4
            sc.polys += box(tx, 1.1, tz, 2.4, 1.4, 1.4, (78, 84, 70), rot=math.radians(6))
            sc.polys += box(tx - 1.5, 0.85, tz, 1.4, 0.9, 1.5, (66, 72, 60))
            for wx in (-0.8, 0.8):
                for wz in (-0.75, 0.75):
                    sc.polys += cylinder(tx + wx, 0.05, tz + wz, 0.32, 0.18, (26, 26, 28), segs=8)
        for k in range(15):
            px = move - 6 - k * 1.15
            pz = 0.6 + (k % 3 - 1) * 0.8
            s = 0.8 if k % 4 else 0.62
            sc.polys += child_figure(px, pz, (72, 74, 78), skin=(150, 120, 96), scale=s,
                                     walk_phase=t * 4.6 + k)
        for k in range(4):
            px = move - 3 - k * 5
            pz = 2.6 if k % 2 else -1.8
            sc.polys += child_figure(px, pz, (52, 56, 48), skin=(140, 112, 90), scale=1.05,
                                     walk_phase=t * 4.6 + k * 2)
        sun_dir = np.array([0.85, 0.18, -0.2]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(70, 60, 96), bg_bot=(210, 130, 78), sun_dir=sun_dir, ambient=0.42)
        draw_glow(img, int(W * 0.12), int(H * 0.5), 300, (255, 150, 70), 1.2)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(ov)
        for sxx in range(0, W, 46):
            dd.polygon([(sxx, H), (sxx + 26, H), (sxx + 210, H * 0.55)], fill=(30, 20, 30, 14))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        add_birds(ImageDraw.Draw(img), t, 4)
        return img
    return render_scene("s6_evening.mp4", frame, N)

# ============================================================================
# SCENE 7 : NIGHT - empty classroom, single sandal pair, candle light
# ============================================================================
def scene_night():
    N = 36 * FPS
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[0.5, 1.15 - 0.35 * prog, 4.6 - 1.9 * prog],
                     yaw=math.radians(4), pitch=math.radians(10 + 8 * prog), fov_focal=880)
        sc.polys += [(np.array([[-7, 0, -3.05], [7, 0, -3.05], [7, 0, 7], [-7, 0, 7]]), (52, 44, 34))]
        sc.polys += box(0, 1.6, -3.1, 14, 3.4, 0.15, (66, 58, 46))
        sc.polys += box(-7.05, 1.6, 2, 0.15, 3.4, 10, (60, 54, 44))
        sc.polys += box(7.05, 1.6, 2, 0.15, 3.4, 10, (56, 50, 40))
        sc.polys += box(0, 3.45, 2, 14, 0.12, 10, (44, 40, 34))
        sc.polys += chalkboard(cy=0)
        sc.polys += school_desk(-3.4, 0, 1.2)
        sc.polys += school_desk(0.2, 0, 2.6)
        sc.polys += school_desk(3.2, 0, 0.6)
        sc.polys += box(4.6, 0.1, 4.4, 0.5, 0.08, 0.5, (70, 54, 34), rot=0.7)
        sc.polys += sandal_pair(0.9, 4.9, rot=0.35, rgb=(146, 88, 50))
        flick = 0.05 * math.sin(t * 21) + 0.03 * math.sin(t * 33)
        sc.polys += cylinder(-3.15, 0.79, 1.05, 0.05, 0.22, (226, 214, 180), segs=8)
        flame = np.array([[-3.15, 1.02, 1.05], [-3.11 + flick, 1.2, 1.05],
                          [-3.15, 1.32 + flick, 1.05], [-3.19 - flick, 1.2, 1.05]])
        sc.polys.append((flame, (255, 190, 90)))
        img = sc.render(bg_top=(6, 8, 16), bg_bot=(12, 12, 18), sun_dir=None,
                        ambient=0.16, draw_bg="black")
        cxp, cyp = 640, 620
        draw_glow(img, cxp, cyp, 260 + 12 * math.sin(t * 18), (255, 170, 80),
                  0.9 + 0.12 * math.sin(t * 22))
        draw_glow(img, 1500, 300, 340, (120, 150, 210), 0.35)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(ov)
        dd.polygon([(1005, 585), (1160, 585), (1240, 900), (960, 900)], fill=(150, 170, 220, 26))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(img)
        random.seed(17)
        for _ in range(30):
            x = random.uniform(500, 900); y = random.uniform(400, 800)
            xx = x + 14 * math.sin(t * 0.6 + y); yy = y - ((t * 9 + x) % 120)
            br = int(90 + 60 * math.sin(t * 2 + x))
            d.ellipse([xx, yy, xx + 2, yy + 2], fill=(min(255, br + 60), min(255, br + 30), br))
        vg = Image.new("L", (W // 4, H // 4), 255)
        dv = ImageDraw.Draw(vg)
        dv.rectangle([8, 6, W // 4 - 8, H // 4 - 6], outline=0, width=6)
        vg = vg.filter(ImageFilter.GaussianBlur(14)).resize((W, H))
        black = Image.new("RGB", (W, H), (0, 0, 0))
        img = Image.composite(img, black, vg.point(lambda p: min(255, p + 120)))
        return img
    return render_scene("s7_night.mp4", frame, N)

# ============================================================================
# SCENE 8 : END - desk with an empty chair; notebook page turns
# ============================================================================
def scene_end_desk():
    N = 26 * FPS
    def frame(i):
        t = i / FPS
        prog = i / N
        sc = Scene3D(cam_pos=[0.4, 1.5, 4.2 - 1.0 * prog], yaw=math.radians(-3),
                     pitch=math.radians(8), fov_focal=880)
        sc.polys += [(np.array([[-7, 0, -3.05], [7, 0, -3.05], [7, 0, 7], [-7, 0, 7]]), (120, 96, 68))]
        sc.polys += box(0, 1.6, -3.1, 14, 3.4, 0.15, (186, 168, 132))
        sc.polys += box(6.95, 1.9, 1.5, 0.06, 1.5, 1.6, (214, 228, 238))
        sc.polys += chalkboard()
        sc.polys += school_desk(0, 0, 0.8)
        sc.polys += box(-0.1, 0.8, 0.75, 0.5, 0.03, 0.36, (244, 242, 234))
        # turning page: rotates up over time then settles
        ang = min(1.0, max(0.0, (prog - 0.25) * 3)) * math.pi * 0.9
        pv = np.array([[-0.35, 0.82, 0.57], [0.35, 0.82, 0.57], [0.35, 0.82, 0.93], [-0.35, 0.82, 0.93]])
        cy_, cz_ = 0.82, 0.93
        pts = []
        for x, y, z in pv:
            dz = z - cz_; dy = y - cy_
            pts.append([x, cy_ + dy * math.cos(ang) - dz * math.sin(ang), cz_ + dy * math.sin(ang) + dz * math.cos(ang)])
        sc.polys.append((np.array(pts), (250, 248, 240)))
        sc.polys += box(0.35, 0.82, 0.9, 0.35, 0.02, 0.04, (210, 160, 60), rot=0.5)
        sun_dir = np.array([0.6, -0.1, -0.5]); sun_dir /= np.linalg.norm(sun_dir)
        img = sc.render(bg_top=(60, 66, 84), bg_bot=(90, 84, 74), sun_dir=sun_dir,
                        ambient=0.55, draw_bg="black")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dd = ImageDraw.Draw(ov)
        dd.polygon([(1420, 330), (1520, 350), (1180, 1080), (980, 1080)], fill=(255, 244, 210, 30))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        return img
    return render_scene("s8_enddesk.mp4", frame, N)

# ============================================================================
# PROCEDURAL AUDIO (synthesized -> copyright free)
# ============================================================================
def save_wav(path, data, rate=44100):
    data = np.clip(data, -1, 1)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(rate)
        wv.writeframes(pcm.tobytes())

def synth_audio():
    rate = 44100
    total_s = 292
    n = total_s * rate
    mix = np.zeros(n)
    tt = np.arange(n) / rate

    def add(seg_start, sig):
        i0 = int(seg_start * rate); i1 = min(n, i0 + len(sig))
        mix[i0:i1] += sig[:i1 - i0]

    def env(a, l):
        e = np.ones(l)
        na = max(1, int(a * rate))
        e[:na] = np.linspace(0, 1, na)
        r = min(na, l); e[-r:] *= np.linspace(1, 0, r)
        return e

    rng = np.random.default_rng(9)
    noise = rng.standard_normal(n)
    wind = np.convolve(noise, np.ones(60) / 60, mode="same") * 0.10
    wind_env = 0.5 + 0.5 * np.sin(tt * 0.11) * np.sin(tt * 0.037)
    add(0, wind * (0.4 + 0.6 * wind_env))

    for b in range(26):
        st = 12 + rng.uniform(0, 50)
        f0 = rng.uniform(2200, 3800)
        dur = rng.uniform(0.08, 0.22)
        m = int(dur * rate)
        tau = np.arange(m) / rate
        freq = f0 + rng.uniform(-600, 900) * np.sin(tau * rng.uniform(20, 45))
        chirp = np.sin(2 * np.pi * np.cumsum(freq) / rate) * np.exp(-tau * 12) * 0.12
        add(st, chirp)

    def bell(st, vol=0.22):
        m = int(3.2 * rate)
        tau = np.arange(m) / rate
        partials = [880, 1174, 1560, 2100, 2640]
        sig = sum(np.sin(2 * np.pi * f * tau) * np.exp(-tau * (2.2 + i)) for i, f in enumerate(partials))
        add(st, sig * env(0.002, m) * vol / 3)
    bell(20.0); bell(61.0, 0.18)

    m = int(85 * rate)
    tau = np.arange(m) / rate
    rn = rng.standard_normal(m)
    lp = np.convolve(rn, np.ones(140) / 140, mode="same")
    thump = 0.5 + 0.5 * np.sin(2 * np.pi * 11.5 * tau) ** 2
    blade = np.sin(2 * np.pi * 23 * tau) * 0.3
    base = (lp * 1.6 + blade) * thump
    ramp = np.clip(np.minimum(tau / 30, (m / rate - tau) / 25), 0, 1) ** 1.5
    add(65, base * ramp * 0.16)

    m = int(50 * rate)
    tau = np.arange(m) / rate
    crk = np.zeros(m)
    for c in range(9):
        fc = rng.uniform(4200, 5200)
        pulses = (np.sin(2 * np.pi * rng.uniform(30, 44) * tau) > 0.86).astype(float)
        crk += np.sin(2 * np.pi * fc * tau) * pulses * rng.uniform(0.012, 0.02)
    add(210, crk)
    for dog in range(4):
        st = 214 + rng.uniform(0, 40)
        md = int(rng.uniform(0.5, 1.1) * rate)
        td = np.arange(md) / rate
        f = 320 - 60 * td
        bark = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-td * 4) * 0.06
        add(st, bark)

    m = int(60 * rate); tau = np.arange(m) / rate
    pad = (np.sin(2 * np.pi * 110 * tau) + 0.6 * np.sin(2 * np.pi * 165 * tau)) * 0.02
    add(150, pad * env(6, m))

    m = int(32 * rate); tau = np.arange(m) / rate
    dr = (np.sin(2 * np.pi * 220 * tau) + 0.5 * np.sin(2 * np.pi * 277 * tau)
          + 0.4 * np.sin(2 * np.pi * 330 * tau)) * 0.03
    add(260, dr * env(4, m))

    mix *= 0.9
    peak = np.max(np.abs(mix)); mix = mix / max(peak, 1e-6) * 0.86
    ap = os.path.join(os.path.dirname(__file__), "audio", "bed.wav")
    save_wav(ap, mix, rate)
    print("audio written:", ap)

# ============================================================================
# BUILD ALL + ASSEMBLE
# ============================================================================
def build():
    S = OUT_DIR
    A = os.path.join(os.path.dirname(__file__), "assets")
    R.text_card(["Since the 2021 coup, thousands of teachers",
                 "refused to work for the military.",
                 "Many villages ran their own schools."],
                os.path.join(S, "c0_topleft.mp4"), 12, sizes=[46, 46, 46])
    R.text_card(["ONE SCHOOL, ONE DAY"], os.path.join(S, "c_title.mp4"), 6,
                sizes=[92], colors=[(240, 220, 160)])
    R.text_card(["A year later, villagers were working to reopen the school.",
                 "In May 2025, in the same township, another school was bombed.",
                 "The UN said twenty students and two teachers were killed."],
                os.path.join(S, "c9_endline.mp4"), 14, sizes=[40, 40, 40])
    R.text_card(["Sources: UNICEF, UN Secretary-General's office, IIMM,",
                 "Myanmar Witness, Reuters, AP, RFA, The Irrawaddy.",
                 "The military says armed rebels were using the school.",
                 "Death tolls vary by source.",
                 "Illustration only. All visuals and audio generated for this film.",
                 "No archive footage used."],
                os.path.join(S, "c10_credits.mp4"), 12,
                sizes=[34, 34, 30, 30, 26, 26],
                colors=[(220, 220, 220)] * 4 + [(150, 150, 150)] * 2)

    R.scene_dawn(); R.scene_classroom(); R.scene_clock(); R.scene_map(); R.scene_helis()
    scene_evening(); scene_night(); scene_end_desk()
    synth_audio()

    ov = lambda src, dst, items: overlay_drawtext(os.path.join(S, src), os.path.join(S, dst), items)
    ov("s1_dawn.mp4", "o1_dawn.mp4", [
        ("Friday, 16 September 2022 - Let Yet Kone, Sagaing Region", 1.5, 7, "(w-text_w)/2", "h-140", 40, "white"),
        ("More than 240 pupils - Myanmar Witness", 9, 6, "(w-text_w)/2", "h-90", 30, "0xDCDCDC"),
        ("ILLUSTRATION", 0.5, 44, "w-text_w-40", "40", 26, "0xB0B0B0")])
    ov("s2_classroom.mp4", "o2_classroom.mp4", [
        ("ILLUSTRATION", 0.5, 39, "w-text_w-40", "40", 26, "0xB0B0B0")])
    ov("s3_clock.mp4", "o3_clock.mp4", [
        ("Around 1:00 pm", 8, 9, "(w-text_w)/2", "h-120", 44, "white")])
    ov("s4_map.mp4", "o4_map.mp4", [
        ("Let Yet Kone - Tabayin Township, Sagaing", 1, 8, "(w-text_w)/2", "80", 40, "white"),
        ("Four helicopters arrive. Two land. Two circle.", 10, 8, "(w-text_w)/2", "h-120", 36, "white"),
        ("About 80 soldiers - residents via RFA", 16, 7, "(w-text_w)/2", "h-80", 28, "0xDCDCDC")])
    ov("s5_helis.mp4", "o5_helis.mp4", [
        ("ILLUSTRATION", 0.5, 29, "w-text_w-40", "40", 26, "0xB0B0B0"),
        ("Firing lasts about one hour - Myanmar Witness", 6, 8, "(w-text_w)/2", "h-120", 38, "white"),
        ("At least 13 people killed - Reuters/AP", 16, 8, "(w-text_w)/2", "h-120", 38, "white"),
        ("UNICEF: at least 11 children", 24, 5, "(w-text_w)/2", "h-80", 30, "0xDCDCDC")])
    ov("s6_evening.mp4", "o6_evening.mp4", [
        ("4:30 pm - soldiers leave with 15 detainees - The Irrawaddy", 2, 9, "(w-text_w)/2", "h-120", 36, "white"),
        ("7 children   5 teachers   3 villagers", 12, 8, "(w-text_w)/2", "h-80", 34, "0xE8C87A"),
        ("The military says armed rebels were using the school.", 21, 8, "(w-text_w)/2", "h-120", 32, "white"),
        ("Villagers deny it. - Reuters / Myanmar Witness", 21, 8, "(w-text_w)/2", "h-80", 28, "0xDCDCDC")])
    ov("s7_night.mp4", "o7_night.mp4", [
        ("Night in Let Yet Kone. A school with no children.", 8, 12, "(w-text_w)/2", "h-110", 40, "0xEDEDED")])
    ov("s8_enddesk.mp4", "o8_enddesk.mp4", [
        ("2022", 3, 6, "(w-text_w)/2", "100", 88, "white"),
        ("2025", 9, 6, "(w-text_w)/2", "100", 88, "0xE8C87A")])

    order = ["c0_topleft", "c_title", "o1_dawn", "o2_classroom", "o3_clock", "o4_map",
             "o5_helis", "o6_evening", "o7_night", "o8_enddesk", "c9_endline", "c10_credits"]
    silent = os.path.join(A, "silent.m4a")
    run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.01", silent])
    lst = os.path.join(A, "list.txt")
    norm = []
    for name in order:
        src = os.path.join(S, name + ".mp4")
        dst = os.path.join(A, name + "_n.mp4")
        run_ffmpeg(["-i", src, "-i", silent, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", dst])
        norm.append(dst)
    with open(lst, "w") as f:
        for p in norm:
            f.write(f"file '{p}'\n")
    merged = os.path.join(S, "video_only.mp4")
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", merged])
    bed = os.path.join(os.path.dirname(__file__), "audio", "bed.wav")
    final = os.path.join(os.path.dirname(__file__), "..", "one_school_one_day_3d.mp4")
    run_ffmpeg(["-i", merged, "-i", bed, "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest",
                "-movflags", "+faststart", final])
    print("FINAL:", final)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "evening": scene_evening()
    elif which == "night": scene_night()
    elif which == "end": scene_end_desk()
    elif which == "audio": synth_audio()
    else: build()
