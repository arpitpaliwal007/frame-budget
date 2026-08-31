"""Frozen video-LLM under test. No training anywhere in this project."""
import torch, numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText

LETTERS = ["A", "B", "C", "D", "E"]


class VLMScorer:
    """Scores a multiple-choice question from a chosen set of frames.

    Design choice that makes the 1-day budget work: instead of free-form
    generation we read the logits of the option letters at the first generated
    position. One forward pass, no decoding loop, fully deterministic -> the
    only variance across runs is the frame subset, which is what we're studying.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = AutoProcessor.from_pretrained(cfg.vlm_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.vlm_id,
            dtype=getattr(torch, cfg.dtype),      # fp16 on a T4; bf16 is NOT supported on sm_75
            attn_implementation=cfg.attn_impl,    # sdpa; flash_attention_2 will fail on Turing
            device_map="cuda",
        ).eval()
        tok = self.proc.tokenizer
        # token id of each bare option letter; verify these are single tokens
        self.letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        assert len(set(self.letter_ids)) == len(LETTERS), "option letters are not distinct tokens"

    @staticmethod
    def _prompt(question, options):
        opts = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, options))
        return (f"{question}\n{opts}\n"
                "Answer with the letter of the correct option only.")

    @torch.inference_mode()
    def score(self, frame_paths, question, options):
        """Returns (probs over options, predicted index)."""
        msgs = [{"role": "user", "content": [
            {"type": "video", "video": [f"file://{p}" for p in frame_paths]},
            {"type": "text", "text": self._prompt(question, options)},
        ]}]
        inputs = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1]            # next-token distribution
        sel = logits[self.letter_ids[:len(options)]].float()   # restrict to valid options
        probs = torch.softmax(sel, -1).cpu().numpy()
        return probs, int(probs.argmax())

    def tokens_per_frame(self, frame_paths, question, options):
        """Measure vision-token cost empirically -- never assume it.
        This number is the unit of the entire cost model."""
        n = len(frame_paths)
        def L(k):
            m = [{"role": "user", "content": [
                {"type": "video", "video": [f"file://{p}" for p in frame_paths[:k]]},
                {"type": "text", "text": self._prompt(question, options)}]}]
            return self.proc.apply_chat_template(
                m, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")["input_ids"].shape[1]
        return (L(n) - L(1)) / max(n - 1, 1)   # slope = marginal tokens per frame


def confidence(probs, kind="maxprob"):
    p = np.sort(probs)[::-1]
    if kind == "maxprob":     return float(p[0])
    if kind == "margin":      return float(p[0] - p[1])
    if kind == "neg_entropy": return float(1 + (probs * np.log(probs + 1e-9)).sum() / np.log(len(probs)))
    raise ValueError(kind)
