"""Prompt / answer templates + timestamp formatting + robust output parsing.

REAL (P0). Used by SFT target construction (P2), reward computation (P7,
`surgground/train/rewards.py`), and generation parsing (P3+, `infer/generate.py`).
Grammar (PLAN.md Appendix B):
    <think> ... </think><answer>[t0, t1]</answer>
    <answer>[[a,b],[c,d]]</answer>        # multi-span
    <answer>ABSTAIN</answer>
    ... optionally trailing "  confidence: NN%"
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# timestamps
# --------------------------------------------------------------------------- #
def format_ts(seconds: float, fmt: str = "seconds") -> str:
    seconds = max(0.0, float(seconds))
    if fmt == "mmss":
        m, s = divmod(int(round(seconds)), 60)
        return f"{m:02d}:{s:02d}"
    return f"{seconds:.1f}"


def parse_ts(s: str) -> float:
    s = s.strip().rstrip("s").strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        out = 0.0
        for p in parts:
            out = out * 60 + p
        return out
    return float(s)


# --------------------------------------------------------------------------- #
# system / user prompts
# --------------------------------------------------------------------------- #
SYSTEM_GROUNDING = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds (total duration given). Answer only "
    "about THIS video. If the asked event does not occur, or the question assumes an "
    "impossible order of events, answer exactly <answer>ABSTAIN</answer>.\n"
    "Think briefly in <think>...</think>, then give the time span as "
    "<answer>[start_seconds, end_seconds]</answer>."
)

SYSTEM_GROUNDING_COARSE = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds (total duration given). Answer only "
    "about THIS video. If the asked event does not occur, or the question assumes an "
    "impossible order, answer exactly <answer>ABSTAIN</answer>.\n"
    "At this step, return the COARSE time window that contains the answer: "
    "<think>brief</think><answer>[win_start_s, win_end_s]</answer>."
)


def render_frame_header(frame_t_sec, duration_s: float, fmt: str = "seconds") -> str:
    ts = ", ".join(format_ts(t, fmt) for t in frame_t_sec)
    return f"Frames sampled at t = [{ts}] s of a {duration_s:.0f} s procedure."


def render_user_grounding(frame_t_sec, duration_s: float, query: str, fmt: str = "seconds") -> str:
    return f"{render_frame_header(frame_t_sec, duration_s, fmt)}\n<video>\n{query}"


def render_answer(spans, abstain: bool = False, think: str | None = None,
                  confidence: int | None = None, fmt: str = "seconds") -> str:
    body = "" if think is None else f"<think>{think}</think>"
    if abstain:
        body += "<answer>ABSTAIN</answer>"
    elif spans and isinstance(spans[0], (list, tuple)):
        inner = ", ".join(f"[{format_ts(a, fmt)}, {format_ts(b, fmt)}]" for a, b in spans)
        body += f"<answer>[{inner}]</answer>"
    else:
        a, b = spans
        body += f"<answer>[{format_ts(a, fmt)}, {format_ts(b, fmt)}]</answer>"
    if confidence is not None:
        body += f" confidence: {int(confidence)}%"
    return body


# --------------------------------------------------------------------------- #
# parsing model output
# --------------------------------------------------------------------------- #
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?::\d+(?:\.\d+)?)*")
_PAIR_RE = re.compile(r"\[\s*([^\[\],]+?)\s*,\s*([^\[\],]+?)\s*\]")
_CONF_RE = re.compile(r"confidence\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


def parse_confidence(text: str) -> float | None:
    m = _CONF_RE.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return max(0.0, min(1.0, v / 100.0 if v > 1.0 else v))


def parse_answer(text: str) -> dict:
    """-> {"kind": "span"|"multi"|"window"|"abstain"|"bad", "spans": [...], "raw": str}.

    "window" and "span" are structurally identical (a single [a,b]); callers that
    prompted for a coarse window treat the result as a window.
    """
    text = text or ""
    m = _ANSWER_RE.search(text)
    inner = m.group(1) if m else text  # lenient: accept a bare payload with no tags
    inner = inner.strip()

    if re.search(r"\bABSTAIN\b", inner, re.IGNORECASE):
        return {"kind": "abstain", "spans": [], "raw": inner}

    pairs = _PAIR_RE.findall(inner)
    spans: list[list[float]] = []
    for a, b in pairs:
        try:
            t0, t1 = parse_ts(a), parse_ts(b)
        except ValueError:
            return {"kind": "bad", "spans": [], "raw": inner}
        if t1 < t0:
            t0, t1 = t1, t0
        spans.append([t0, t1])

    if not spans:
        return {"kind": "bad", "spans": [], "raw": inner}
    if len(spans) == 1:
        return {"kind": "span", "spans": spans, "raw": inner}
    return {"kind": "multi", "spans": spans, "raw": inner}


def parse_span(text: str):
    """First [a, b] span as (float, float), or None."""
    p = parse_answer(text)
    return tuple(p["spans"][0]) if p["spans"] else None
