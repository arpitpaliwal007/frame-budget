"""SigLIP embeddings for every frame + every question.

~45k frames on a T4 takes about 4 minutes. After this, ALL frame selection is
pure numpy on cached arrays -- sweeping 4 selectors x 6 budgets costs 0 GPU.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from config import CFG

dev = "cuda"
proc = AutoProcessor.from_pretrained(CFG.retriever_id)
model = AutoModel.from_pretrained(CFG.retriever_id, dtype=torch.float16).to(dev).eval()
os.makedirs(CFG.feat_dir, exist_ok=True)
rows = json.load(open(CFG.evalset))

# ---- frames ----
for n, v in enumerate(sorted({r["video"] for r in rows})):
    dst = f"{CFG.feat_dir}/{v}.npy"
    if os.path.exists(dst):
        continue
    paths = sorted(glob.glob(f"{CFG.frames_dir}/{v}/*.jpg"))
    if not paths:
        continue
    embs = []
    for i in range(0, len(paths), 64):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
        px = proc(images=ims, return_tensors="pt").to(dev)
        with torch.inference_mode():
            e = model.get_image_features(pixel_values=px["pixel_values"].half())
        embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    np.save(dst, np.concatenate(embs))
    if n % 100 == 0:
        print("videos done:", n, flush=True)

# ---- questions ----
qs = [r["question"] for r in rows]
tv = []
for i in range(0, len(qs), 128):
    tk = proc(text=qs[i:i + 128], padding="max_length", truncation=True, return_tensors="pt").to(dev)
    with torch.inference_mode():
        e = model.get_text_features(**tk)
    tv.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
np.save(f"{CFG.feat_dir}/_questions.npy", np.concatenate(tv))
json.dump([r["qid"] for r in rows], open(f"{CFG.feat_dir}/_qids.json", "w"))
print("features cached ->", CFG.feat_dir)
