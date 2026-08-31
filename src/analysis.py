"""Statistics that separate an Applied Scientist writeup from a hackathon demo.

Every accuracy number gets a CI. Every method comparison gets a PAIRED test
(same questions, same model -> McNemar, not a two-sample t-test). Every
confidence signal gets a calibration and a risk-coverage curve.
"""
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- accuracy
def bootstrap_ci(correct, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on accuracy. correct: 0/1 array."""
    c = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), size=(n_boot, len(c)))
    boots = c[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return c.mean(), lo, hi


def mcnemar(a, b):
    """Exact McNemar on two PAIRED correctness vectors over identical questions.

    b01 = a wrong & b right, b10 = a right & b wrong.
    Returns (p_value, b01, b10, delta_acc). Report the effect size, not just p.
    """
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    b01 = int((~a & b).sum())
    b10 = int((a & ~b).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else float(stats.binomtest(b01, n, 0.5).pvalue)
    return p, b01, b10, float(b.mean() - a.mean())


# ------------------------------------------------------------- calibration
def ece(conf, correct, n_bins=10):
    """Expected Calibration Error. Does 'the model is 90% sure' mean 90% right?"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def risk_coverage(conf, correct):
    """Selective prediction curve: if we only answer the top-c fraction by
    confidence, what's the error rate? AURC low = the signal is a good gate.
    This is the direct evidence that the cascade's trigger is trustworthy."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    err = 1 - correct[order]
    cov = np.arange(1, len(err) + 1) / len(err)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return cov, risk, float(risk.mean())


# ------------------------------------------------------------------ cascade
def cascade_curve(conf_lo, correct_lo, correct_hi, frames_lo, frames_hi, n_tau=60):
    """Sweep the escalation threshold. Returns (avg_frames, accuracy, tau).

    Cheap pass at frames_lo; escalate the least-confident questions to frames_hi.
    Costs ZERO extra GPU -- it replays cached predictions from the fixed-K runs.
    """
    conf = np.asarray(conf_lo, float)
    lo, hi = np.asarray(correct_lo, float), np.asarray(correct_hi, float)
    out = []
    for tau in np.quantile(conf, np.linspace(0, 1, n_tau)):
        esc = conf < tau
        acc = np.where(esc, hi, lo).mean()
        # honest accounting: an escalated query pays for BOTH passes
        frames = np.where(esc, frames_lo + frames_hi, frames_lo).mean()
        out.append((frames, acc, float(tau), float(esc.mean())))
    return np.array([(f, a) for f, a, _, _ in out]), out


def oracle_routing(correct_lo, correct_hi, frames_lo, frames_hi):
    """Upper bound: escalate ONLY the questions that flip wrong -> right.
    The gap between this and the achieved curve is your remaining headroom."""
    lo, hi = np.asarray(correct_lo, bool), np.asarray(correct_hi, bool)
    flip = ~lo & hi
    acc = np.where(flip, hi, lo).mean()
    frames = np.where(flip, frames_lo + frames_hi, frames_lo).mean()
    return float(frames), float(acc), float(flip.mean())


def pareto_front(points):
    """Keep only (cost, acc) pairs not dominated by a cheaper-and-better one."""
    pts = sorted(points, key=lambda p: p[0])
    front, best = [], -np.inf
    for c, a in pts:
        if a > best:
            front.append((c, a)); best = a
    return front


# ----------------------------------------------------------------- cost
def cost_report(avg_frames, tokens_per_frame, usd_per_1k_tokens, n_queries=1_000_000):
    tok = avg_frames * tokens_per_frame
    return {"avg_frames": avg_frames, "vision_tokens_per_query": tok,
            "usd_per_1M_queries": tok * n_queries / 1000 * usd_per_1k_tokens}
