"""Decode every needed video to 1-fps JPEGs. ffmpeg, parallel, resumable.

Do this ONCE and persist to Drive. After this the raw videos are never touched
again, which is what makes a Colab disconnect survivable.
"""
import sys, os, json, subprocess, glob
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import CFG

VIDEO_ROOT = os.environ["FB_VIDEO_ROOT"]     # where the NExT-QA .mp4s live
rows = json.load(open(CFG.evalset))
videos = sorted({r["video"] for r in rows})

def do(v):
    out = f"{CFG.frames_dir}/{v}"
    if len(glob.glob(f"{out}/*.jpg")) > 0:
        return "skip"
    os.makedirs(out, exist_ok=True)
    src = next(iter(glob.glob(f"{VIDEO_ROOT}/**/{v}.mp4", recursive=True)), None)
    if src is None:
        return f"MISSING {v}"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src,
                    "-vf", f"fps={CFG.fps},scale=-2:224", "-q:v", "3",
                    "-frames:v", str(CFG.max_frames_pool), f"{out}/%04d.jpg"], check=False)
    return "ok"

with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(do, videos))
print({k: res.count(k) for k in set(r.split()[0] for r in res)})
print("missing:", [r for r in res if r.startswith("MISSING")][:10])
