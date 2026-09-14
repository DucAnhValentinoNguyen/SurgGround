"""Video + chat collator per backend (PLAN.md 8.5, P2 helpers / P4 full path).

P2 delivers the **pure, model-free** pieces that task construction and the P4 SFT
recipe both need:

  * ``plan_micro_batches`` — split a frame list into fixed chunks so the frozen
    vision tower is encoded in bounded-VRAM micro-batches (8 frames/chunk).
  * ``expand_media_tokens`` — expand a single ``<image>`` / ``<video>`` placeholder
    into the exact number of vision tokens the connector will emit, per backend.
  * ``assistant_char_span`` — the ``<answer>`` region to keep unmasked in the loss.
  * ``build_plain_chat`` — a deterministic fallback chat rendering for tests.

The real ``Collator.__call__`` (tokenizer + image processor + pixel loading from
``$FRAMES_ROOT``) lands in **P4** together with ``models/base.py``.
"""
from __future__ import annotations

_MEDIA_TOKEN = {
    "internvl3_2b": "<image>",
    "internvl3_8b": "<image>",
    "llava_video_7b": "<image>",
    "qwen25vl_7b": "<|vision_start|><|image_pad|><|vision_end|>",
}


def plan_micro_batches(n_frames: int, chunk: int = 8) -> list[tuple[int, int]]:
    """[(start, end), ...] half-open index ranges of at most ``chunk`` frames."""
    if n_frames <= 0:
        return []
    chunk = max(1, int(chunk))
    return [(i, min(i + chunk, n_frames)) for i in range(0, n_frames, chunk)]


def n_vision_tokens(n_frames: int, tokens_per_frame: int, temporal_stride: int = 1,
                    spatial_pool: int = 1) -> int:
    """Post-connector vision-token count (STORM reduction): frames are strided by
    ``temporal_stride`` on the time axis and each frame's grid is pooled by
    ``spatial_pool`` on each spatial side."""
    import math

    t = math.ceil(n_frames / max(1, temporal_stride))
    per = max(1, tokens_per_frame // max(1, spatial_pool * spatial_pool))
    return t * per


def expand_media_tokens(text: str, n_tokens: int, backend: str = "internvl3_2b") -> str:
    """Replace the first ``<image>``/``<video>`` marker in ``text`` with ``n_tokens``
    copies of the backend media token (the connector output count must match)."""
    tok = _MEDIA_TOKEN.get(backend, "<image>")
    filled = tok * max(0, int(n_tokens))
    for marker in ("<video>", "<image>"):
        if marker in text:
            return text.replace(marker, filled, 1)
    return text


def assistant_char_span(messages) -> tuple[int, int]:
    """Character offsets of the assistant turn's content inside ``build_plain_chat``
    output — the span that stays unmasked in the SFT loss."""
    rendered = build_plain_chat(messages, add_generation_prompt=False)
    for m in messages:
        if m.get("role") == "assistant":
            start = rendered.find(m["content"])
            return (start, start + len(m["content"])) if start >= 0 else (0, 0)
    return (0, 0)


def build_plain_chat(messages, add_generation_prompt: bool = True) -> str:
    """Backend-agnostic deterministic chat rendering (tests + a last-resort path
    when a backend has no chat template)."""
    parts = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    return "\n".join(parts)


class Collator:
    """Full video + chat collation for one backend. Pixel/tokenizer path is P4."""

    def __init__(self, processor, cfg, backend: str):
        self.processor = processor
        self.cfg = cfg
        self.backend = backend

    def __call__(self, batch):
        raise NotImplementedError(
            "P4: needs a loaded processor + frame pixels from $FRAMES_ROOT; "
            "the model-free helpers in this module are the P2 deliverable."
        )
