"""Turn the JSONL into the figures and the numbers you will defend."""
import sys, os, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CFG
import analysis as A

TPF = float(os.environ.get("FB_TOKENS_PER_FRAME", 64))   # measure with vlm.tokens_per_frame

R = [json.loads(l) for l in open(f"{CFG.runs_dir}/main.jsonl")]
by = collections.defaultdict(dict)
for r in R:
    by[(r["selector"], r["k"])][r["qid"]] = r
qids = sorted(set.intersection(*[set(v) for v in by.values()]))   # common support only
print(f"{len(R)} results | {len(by)} cells | {len(qids)} questions on common support\n")

vec = lambda s, k, f="correct": np.array([by[(s, k)][q][f] for q in qids])

# ---- 1. fixed-budget Pareto -------------------------------------------------
print("=== accuracy vs frame budget (95% bootstrap CI) ===")
tab = {}
for (s, k) in sorted(by):
    m, lo, hi = A.bootstrap_ci(vec(s, k))
    tab[(s, k)] = m
    print(f"  {s:9s} k={k:<3d} acc={m:.4f}  [{lo:.4f}, {hi:.4f}]")

# ---- 2. does query-aware selection actually beat uniform? -------------------
print("\n=== paired McNemar vs uniform at matched budget ===")
for k in sorted({k for _, k in by}):
    if ("uniform", k) not in by:
        continue
    for s in [s for s, kk in by if kk == k and s != "uniform"]:
        p, b01, b10, d = A.mcnemar(vec("uniform", k), vec(s, k))
        flag = "SIG" if p < 0.05 else "n.s."
        print(f"  k={k:<3d} {s:9s} delta={d:+.4f}  p={p:.3g}  ({b01} gained / {b10} lost)  {flag}")

# ---- 3. the cascade ---------------------------------------------------------
print("\n=== confidence-gated cascade ===")
best = max([s for s, k in by if k == CFG.k_hi], key=lambda s: tab[(s, CFG.k_hi)])
lo_c, hi_c = vec("uniform", CFG.k_lo), vec(best, CFG.k_hi)
for sig in ["maxprob", "margin", "neg_entropy"]:
    conf = vec("uniform", CFG.k_lo, sig)
    _, det = A.cascade_curve(conf, lo_c, hi_c, CFG.k_lo, CFG.k_hi)
    hit = [d for d in det if d[1] >= hi_c.mean() - 1e-9]
    cheapest = min(hit, key=lambda d: d[0]) if hit else max(det, key=lambda d: d[1])
    print(f"  {sig:12s} ECE={A.ece(conf, lo_c):.4f} AURC={A.risk_coverage(conf, lo_c)[2]:.4f} "
          f"| matches uniform-{CFG.k_hi} at {cheapest[0]:.2f} avg frames (esc {cheapest[3]:.1%})")
of, oa, fr = A.oracle_routing(lo_c, hi_c, CFG.k_lo, CFG.k_hi)
print(f"  ORACLE routing: {oa:.4f} @ {of:.2f} avg frames ({fr:.1%} of questions flip)")

# ---- 4. where does it help? -------------------------------------------------
print("\n=== accuracy by question type (uniform-4 -> uniform-16) ===")
types = collections.defaultdict(list)
for i, q in enumerate(qids):
    types[by[("uniform", CFG.k_lo)][q]["type"]].append(i)
for t, ix in sorted(types.items()):
    print(f"  {t:6s} n={len(ix):4d}  k={CFG.k_lo}: {lo_c[ix].mean():.3f}  k={CFG.k_hi}: {hi_c[ix].mean():.3f}  "
          f"delta={hi_c[ix].mean()-lo_c[ix].mean():+.3f}")

# ---- 5. plot ----------------------------------------------------------------
os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for s in sorted({s for s, _ in by}):
    ks = sorted(k for ss, k in by if ss == s)
    ax.plot([k * TPF for k in ks], [tab[(s, k)] for k in ks], "o-", label=s)
conf = vec("uniform", CFG.k_lo, "margin")
_, det = A.cascade_curve(conf, lo_c, hi_c, CFG.k_lo, CFG.k_hi)
ax.plot([d[0] * TPF for d in det], [d[1] for d in det], "k--", lw=2, label="cascade (ours)")
ax.set_xscale("log"); ax.set_xlabel("vision tokens per query"); ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs inference cost"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/pareto.png", dpi=160)
print("\nwrote results/pareto.png")
