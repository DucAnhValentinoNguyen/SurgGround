"""Efficiency table (PLAN.md 9.6) — the "runs on 24 GB" evidence.

Per regime: #frames, #vision tokens (post-connector), LLM context length,
prefill / decode latency, tokens/s, **peak VRAM**, GPU-seconds/query.

P2 delivers the model-free arithmetic + the timing harness; ``measure`` runs it
against a loaded backend in P9 (``torch.cuda.synchronize`` timers, 20-query
warm-up + 100-query measure on the fixed 4090).
"""
from __future__ import annotations

import time

import numpy as np


# --------------------------------------------------------------------------- #
# STORM token arithmetic
# --------------------------------------------------------------------------- #
def count_vision_tokens(n_frames: int, tokens_per_frame: int,
                        temporal_stride: int = 1, spatial_pool: int = 1) -> int:
    """Post-connector vision-token count. ``temporal_stride`` keeps 1 frame every
    N on the time axis; ``spatial_pool`` pools each frame's token grid by S on
    each side (so S*S fewer tokens/frame)."""
    import math

    t = math.ceil(n_frames / max(1, temporal_stride))
    per = max(1, tokens_per_frame // max(1, spatial_pool * spatial_pool))
    return t * per


def context_length(n_vision_tokens: int, n_text_tokens: int) -> int:
    return int(n_vision_tokens) + int(n_text_tokens)


def tokens_per_video(n_frames_by_regime: dict, tokens_per_frame: int,
                     temporal_stride: int = 1, spatial_pool: int = 1) -> dict:
    return {r: count_vision_tokens(nf, tokens_per_frame, temporal_stride, spatial_pool)
            for r, nf in n_frames_by_regime.items()}


# --------------------------------------------------------------------------- #
# timing harness
# --------------------------------------------------------------------------- #
class EfficiencyTimer:
    """Context manager: wall time (ms) + peak CUDA memory (GB) for the block.

    ``vram_gb`` is ``None`` on CPU. Uses ``torch.cuda.synchronize`` and resets the
    peak-memory counter on enter so the measurement is isolated.
    """

    def __init__(self):
        self.ms = None
        self.vram_gb = None
        self._t0 = None
        self._torch = None

    def __enter__(self):
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
        except Exception:
            self._torch = None
        if self._torch is not None:
            self._torch.cuda.synchronize()
            self._torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._torch is not None:
            self._torch.cuda.synchronize()
            self.vram_gb = self._torch.cuda.max_memory_allocated() / 2**30
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False


def summarize(runs) -> dict:
    """``runs``: per-query dicts with any of ``prefill_ms``, ``decode_ms``,
    ``n_new_tokens``, ``n_frames``, ``n_vis_tokens``, ``ctx_len``, ``peak_vram_gb``.
    Returns the means / maxes for the ``long_results.csv`` efficiency columns."""
    runs = list(runs)
    if not runs:
        return {"n": 0}

    def col(k):
        return np.asarray([r[k] for r in runs if r.get(k) is not None], dtype=float)

    prefill, decode = col("prefill_ms"), col("decode_ms")
    new_tok = col("n_new_tokens")
    vram = col("peak_vram_gb")
    tok_s = float(new_tok.sum() / (decode.sum() / 1000.0)) if decode.sum() > 0 else float("nan")
    total_ms = (prefill.sum() if prefill.size else 0.0) + (decode.sum() if decode.size else 0.0)
    out = {
        "prefill_ms": float(np.mean(prefill)) if prefill.size else float("nan"),
        "decode_ms": float(np.mean(decode)) if decode.size else float("nan"),
        "tokens_s": tok_s,
        "peak_vram_gb": float(np.max(vram)) if vram.size else None,
        "gpu_s_per_q": float(total_ms / 1000.0 / len(runs)) if total_ms else float("nan"),
        "n": len(runs),
    }
    for k in ("n_frames", "n_vis_tokens", "ctx_len"):
        c = col(k)
        if c.size:
            out[k] = float(np.mean(c))
    return out


def measure(model, queries, cfg):
    raise NotImplementedError(
        "P9: run warmup (cfg.eval.efficiency.warmup_queries) then measure "
        "(measure_queries) with EfficiencyTimer around prefill + decode, "
        "then summarize(runs)."
    )
