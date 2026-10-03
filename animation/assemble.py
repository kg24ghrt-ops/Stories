#!/usr/bin/env python3
"""
Assemble the complete film from the scene clips in animation/scenes/.

Story order (English narration script, shot list 0:00 -> 5:00):
  c0_topleft   TOP LINE card          (complete)
  c_title      Title card             (complete)
  s1_dawn      MORNING - dawn         (complete)
  s2_classroom MORNING - classroom   (MISSING -> freeze first frame as gap filler)
  s3_clock     AFTERNOON - clock 1pm  (complete)
  s4_map       AFTERNOON - map        (MISSING -> freeze first frame as gap filler)
  s5_helis     AFTERNOON - helicopters(MISSING -> freeze first frame as gap filler)
  s6_evening   EVENING                (MISSING -> freeze first frame as gap filler)
  s7_night     NIGHT                  (MISSING -> freeze first frame as gap filler)
  s8_enddesk   END desk 2022->2025    (MISSING -> freeze first frame as gap filler)
  c9_endline   END LINE card          (complete)
  c10_credits  CREDITS card           (complete)

For every missing/incomplete scene (duration < 2s) we extract its real first
frame and loop it into a proper clip with ffmpeg, so nothing is skipped.
Then all clips are normalized (same codec/fps/audio) and concatenated.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "scenes")
A = os.path.join(HERE, "assets")
os.makedirs(A, exist_ok=True)

ORDER = [
    ("c0_topleft", 12),
    ("c_title",     6),
    ("s1_dawn",    16),
    ("s2_classroom", 20),   # target duration when repairing a gap
    ("s3_clock",   18),
    ("s4_map",     18),
    ("s5_helis",   20),
    ("s6_evening", 30),
    ("s7_night",   26),
    ("s8_enddesk", 16),
    ("c9_endline", 14),
    ("c10_credits", 12),
]

MIN_GOOD = 2.0  # anything shorter than this counts as a missing gap


def ffprobe_dur(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def run(args):
    print("+ ffmpeg " + " ".join(args[:12]) + (" ..." if len(args) > 12 else ""))
    r = subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        sys.exit(f"ffmpeg failed: {' '.join(args)}")


def repair_gap(name, target_dur):
    """Extract first frame of a stub clip and loop it into a real clip."""
    src = os.path.join(S, name + ".mp4")
    png = os.path.join(S, name + "_first.png")
    dst = os.path.join(S, name + ".mp4.tmp.mp4")
    # 1) grab the true first frame
    run(["-i", src, "-frames:v", "1", png])
    # 2) loop it as video with a slow Ken-Burns push-in so held frames breathe
    run(["-loop", "1", "-framerate", "30", "-t", str(target_dur), "-i", png,
         "-vf", ("scale=2112:1188,zoompan=z='min(1.12,1+0.0009*on)':"
                 "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'d=1:"
                 f"s=1920x1080:s=30,format=yuv420p,fps=30,"
                 f"fade=t=in:st=0:d=1.2,fade=t=out:st={target_dur-1.5}:d=1.5"),
         "-c:v", "libx264", "-preset", "medium", "-crf", "19", dst])
    os.replace(dst, src)
    print(f"repaired gap: {name} -> {target_dur}s held frame")


def main():
    # --- Step 1: fill missing gaps with held first frames -------------------
    for name, target in ORDER:
        src = os.path.join(S, name + ".mp4")
        if not os.path.exists(src):
            sys.exit(f"scene file missing entirely: {src}")
        d = ffprobe_dur(src)
        if d < MIN_GOOD:
            repair_gap(name, target)

    # --- Step 2: silent audio track so concat demuxer works uniformly -------
    silent = os.path.join(A, "silent.m4a")
    run(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.01", silent])

    norm = []
    for name, target in ORDER:
        src = os.path.join(S, name + ".mp4")
        dst = os.path.join(A, name + "_n.mp4")
        dur = ffprobe_dur(src)
        run(["-i", src, "-i", silent,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
             "-r", "30", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ar", "44100", "-ac", "1",
             "-af", "apad", "-t", str(dur), dst])
        norm.append(dst)

    lst = os.path.join(A, "list.txt")
    with open(lst, "w") as f:
        for p in norm:
            f.write(f"file '{p}'\n")

    merged = os.path.join(S, "video_only.mp4")
    run(["-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", merged])

    final = os.path.join(HERE, "..", "one_school_one_day_final.mp4")
    run(["-i", merged, "-map", "0:v", "-map", "0:a",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", final])

    d = ffprobe_dur(final)
    print(f"DONE: {os.path.abspath(final)}  ({d:.1f}s = {d/60:.2f} min)")


if __name__ == "__main__":
    main()
