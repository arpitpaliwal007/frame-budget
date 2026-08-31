"""Freeze a stratified eval set ONCE. Never resample it after you see results."""
import sys, json, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from datasets import load_dataset
from config import CFG

ds = load_dataset(CFG.dataset_id, CFG.dataset_cfg, split="test")
rng = np.random.default_rng(CFG.seed)

by_type = collections.defaultdict(list)
for i, r in enumerate(ds):
    by_type[r["type"]].append(i)

per = CFG.n_questions // len(by_type)
picked = []
for t, idxs in sorted(by_type.items()):
    take = min(per, len(idxs))
    picked += rng.choice(idxs, size=take, replace=False).tolist()
    print(f"{t:6s} pool={len(idxs):6d} take={take}")

rows = [{"qid": f"{ds[i]['video']}_{ds[i]['qid']}", "video": str(ds[i]["video"]),
         "question": ds[i]["question"], "type": ds[i]["type"],
         "options": [ds[i][f"a{j}"] for j in range(5)], "answer": int(ds[i]["answer"])}
        for i in sorted(picked)]

os.makedirs(CFG.root, exist_ok=True)
json.dump(rows, open(CFG.evalset, "w"), indent=1)
print(f"\nwrote {len(rows)} questions over {len({r['video'] for r in rows})} videos -> {CFG.evalset}")
print("SANITY: answer distribution", collections.Counter(r["answer"] for r in rows))
