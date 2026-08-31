"""Frame selectors. All operate on cached SigLIP features -> pure numpy, zero GPU.

# NOTE: named frame_selectors, NOT selectors -- "selectors" shadows a Python
# stdlib module that subprocess imports, and breaks scipy/torch imports.

This is the reason the whole study fits in a day: once features are cached,
sweeping selectors and budgets costs nothing but the VLM forward passes.
"""
import numpy as np


def uniform(n_frames: int, k: int, **_) -> np.ndarray:
    """Evenly spaced. The standard baseline every video-LLM paper uses."""
    if k >= n_frames:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, k).round().astype(int)


def random_sel(n_frames: int, k: int, rng: np.random.Generator = None, **_) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    k = min(k, n_frames)
    return np.sort(rng.choice(n_frames, size=k, replace=False))


def topk_sim(n_frames: int, k: int, sim: np.ndarray = None, **_) -> np.ndarray:
    """Naive query-aware: top-k by SigLIP text-image similarity.

    Known failure mode -> picks k near-duplicate frames from one moment and
    throws away all temporal coverage. That failure is exactly what `mmr` fixes,
    and showing it empirically is half the story of the report.
    """
    k = min(k, n_frames)
    return np.sort(np.argsort(-sim)[:k])


def mmr(n_frames: int, k: int, sim: np.ndarray = None, lam: float = 0.6,
        feats: np.ndarray = None, **_) -> np.ndarray:
    """Greedy submodular selection: relevance + coverage.

    Maximises  f(S) = lam * sum_{i in S} s_i + (1-lam) * FacilityLocation(S)
    with FacilityLocation(S) = sum_j max_{i in S} sim(frame_i, frame_j).

    f is monotone submodular, so greedy is (1 - 1/e)-optimal. That guarantee is
    the thing to say on a whiteboard when an interviewer asks "why greedy?".
    This is our reimplementation of the AKS / Q-Frame family of selectors.
    """
    k = min(k, n_frames)
    rng_span = np.ptp(sim) + 1e-8
    s = (sim - sim.min()) / rng_span
    F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    S_ij = F @ F.T                                  # (n, n) frame-frame similarity
    cov = np.zeros(n_frames)                        # coverage after picking S
    picked = []
    for _ in range(k):
        new_cov = np.maximum(cov[None, :], S_ij)    # (n_cand, n) coverage if we add cand
        gains = lam * s + (1 - lam) * (new_cov.sum(1) - cov.sum())
        gains[picked] = -np.inf
        c = int(np.argmax(gains))
        picked.append(c)
        cov = new_cov[c]
    return np.sort(np.array(picked))


REGISTRY = {"uniform": uniform, "random": random_sel, "topk_sim": topk_sim, "mmr": mmr}


def select(name, n_frames, k, sim=None, feats=None, rng=None, lam=0.6):
    return REGISTRY[name](n_frames=n_frames, k=k, sim=sim, feats=feats, rng=rng, lam=lam)
