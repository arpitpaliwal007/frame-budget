"""The main sweep. Append-only JSONL, keyed and resumable.

Colab WILL disconnect on you. Every result is flushed the moment it exists and
re-running skips whatever is already on disk. Treat this as non-negotiable
infrastructure, not polish -- it is the difference between finishing and not.

Usage:
  python scripts/04_run_eval.py --selectors uniform --budgets 1 2 4 8 16 32
  python scripts/04_run_eval.py --selectors topk_sim mmr random --budgets 2 4 8 16
"""
import sys, os, json, glob, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from config import CFG
import frame_selectors as FS
from vlm import VLMScorer, confidence

ap = argparse.ArgumentParser()
ap.add_argument("--selectors", nargs="+", default=list(CFG.selectors))
ap.add_argument("--budgets", nargs="+", type=int, default=list(CFG.budgets))
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
A = ap.parse_args()

os.makedirs(CFG.runs_dir, exist_ok=True)
OUT = A.out or f"{CFG.runs_dir}/main.jsonl"

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line); done.add((r["qid"], r["selector"], r["k"]))
        except Exception:
            pass                      # tolerate a half-written last line from a hard kill
print(f"resuming: {len(done)} results already on disk")

rows = json.load(open(CFG.evalset))[: A.limit]
qids = json.load(open(f"{CFG.feat_dir}/_qids.json"))
qvec = np.load(f"{CFG.feat_dir}/_questions.npy")
qpos = {q: i for i, q in enumerate(qids)}

vlm = VLMScorer(CFG)
rng = np.random.default_rng(CFG.seed)
fh = open(OUT, "a", buffering=1)       # line-buffered: survives a kill -9
t0, n = time.time(), 0

for r in rows:
    paths = sorted(glob.glob(f"{CFG.frames_dir}/{r['video']}/*.jpg"))
    if not paths:
        continue
    feats = np.load(f"{CFG.feat_dir}/{r['video']}.npy")[: len(paths)]
    sim = feats @ qvec[qpos[r["qid"]]]
    for sel in A.selectors:
        for k in A.budgets:
            if (r["qid"], sel, k) in done:
                continue
            idx = FS.select(sel, len(paths), k, sim=sim, feats=feats,
                            rng=rng, lam=CFG.mmr_lambda)
            probs, pred = vlm.score([paths[i] for i in idx], r["question"], r["options"])
            fh.write(json.dumps({
                "qid": r["qid"], "type": r["type"], "selector": sel, "k": int(k),
                "n_pool": len(paths), "frames": [int(i) for i in idx],
                "pred": pred, "answer": r["answer"], "correct": int(pred == r["answer"]),
                "probs": [round(float(p), 5) for p in probs],
                "maxprob": confidence(probs, "maxprob"),
                "margin": confidence(probs, "margin"),
                "neg_entropy": confidence(probs, "neg_entropy"),
            }) + "\n")
            n += 1
            if n % 200 == 0:
                r_ = n / (time.time() - t0)
                print(f"{n} calls | {r_:.2f} it/s | eta {(len(rows)*len(A.selectors)*len(A.budgets)-n)/r_/60:.0f} min", flush=True)
print("done:", n, "new calls in", f"{(time.time()-t0)/60:.1f} min")
