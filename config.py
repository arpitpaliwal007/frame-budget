"""Central config. Everything that could change an experimental result lives here."""
import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Cfg:
    # --- paths (point ROOT at Google Drive so a Colab disconnect doesn't cost you the day) ---
    root: str = os.environ.get("FB_ROOT", "/content/drive/MyDrive/frame-budget")

    # --- system under test ---
    vlm_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    retriever_id: str = "google/siglip-base-patch16-224"
    dtype: str = "float16"          # T4 is Turing (sm_75): NO bf16, NO FlashAttention-2.
    attn_impl: str = "sdpa"         # do not pass flash_attention_2 on a T4, it will fail.

    # --- eval set ---
    dataset_id: str = "lmms-lab/NExTQA"
    dataset_cfg: str = "MC"
    n_questions: int = 1200         # FALLBACK: drop to 600 if H4 throughput < 1.5 it/s
    seed: int = 1234

    # --- experiment grid ---
    budgets: tuple = (1, 2, 4, 8, 16, 32)
    selectors: tuple = ("uniform", "random", "topk_sim", "mmr")
    mmr_lambda: float = 0.6         # relevance vs temporal-coverage tradeoff
    fps: float = 1.0                # decode rate
    max_frames_pool: int = 64       # candidate pool per video

    # --- cascade ---
    k_lo: int = 4
    k_hi: int = 16
    conf_signals: tuple = ("maxprob", "margin", "neg_entropy", "two_view_agree")

    # --- cost model (vision tokens measured empirically, not assumed) ---
    usd_per_1k_input_tokens: float = 0.0  # fill from a real price sheet; keep the source in the README

    @property
    def frames_dir(self): return f"{self.root}/frames"
    @property
    def feat_dir(self):   return f"{self.root}/features"
    @property
    def runs_dir(self):   return f"{self.root}/runs"
    @property
    def evalset(self):    return f"{self.root}/eval_set.json"

CFG = Cfg()
