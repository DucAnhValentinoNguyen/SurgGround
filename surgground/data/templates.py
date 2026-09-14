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


# --------------------------------------------------------------------------- #
# phase / step segmentation (T2)  ·  grammar: <answer>[["P3", 612, 1804], ...]</answer>
# --------------------------------------------------------------------------- #
SYSTEM_PHASE = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds (total duration given). Segment the "
    "whole procedure into contiguous phases in order. Think briefly in <think>...</think>, "
    'then answer <answer>[["<label>", start_seconds, end_seconds], ...]</answer>.'
)
SYSTEM_STEP = SYSTEM_PHASE.replace("into contiguous phases", "into contiguous steps")

_SEG_RE = re.compile(
    r"\[\s*\"?([A-Za-z0-9_+\- ]+?)\"?\s*,\s*([\d.:]+)\s*,\s*([\d.:]+)\s*\]"
)


def render_user_segmentation(frame_t_sec, duration_s: float, unit: str = "phases",
                             fmt: str = "seconds") -> str:
    return (f"{render_frame_header(frame_t_sec, duration_s, fmt)}\n<video>\n"
            f"Segment this operation into {unit}.")


def render_answer_segments(segments, think: str | None = None, fmt: str = "seconds") -> str:
    """`segments`: list of (label, t0, t1)."""
    body = "" if think is None else f"<think>{think}</think>"
    inner = ", ".join(
        f'["{label}", {format_ts(t0, fmt)}, {format_ts(t1, fmt)}]' for label, t0, t1 in segments
    )
    return body + f"<answer>[{inner}]</answer>"


def parse_segments(text: str) -> list[tuple[str, float, float]]:
    """-> [(label, t0, t1), ...] parsed from an <answer> segment list (lenient)."""
    text = text or ""
    m = _ANSWER_RE.search(text)
    inner = m.group(1) if m else text
    out: list[tuple[str, float, float]] = []
    for label, a, b in _SEG_RE.findall(inner):
        try:
            t0, t1 = parse_ts(a), parse_ts(b)
        except ValueError:
            continue
        if t1 < t0:
            t0, t1 = t1, t0
        out.append((label.strip(), t0, t1))
    return out


# --------------------------------------------------------------------------- #
# remaining surgery duration (T3)  ·  grammar: <answer>42.0</answer>  (minutes)
# --------------------------------------------------------------------------- #
SYSTEM_RSD = (
    "You are a surgical video assistant. You see frames sampled from the START of an "
    "operation up to the current moment (timestamps in seconds). Estimate how many "
    "minutes of surgery REMAIN after the last frame. Think briefly in <think>...</think>, "
    "then answer <answer>MINUTES</answer> as a single number."
)

_MIN_RE = re.compile(r"-?\d+(?:\.\d+)?")


def render_user_rsd(frame_t_sec, elapsed_s: float, fmt: str = "seconds") -> str:
    hdr = (f"Frames sampled at t = "
           f"[{', '.join(format_ts(t, fmt) for t in frame_t_sec)}] s; "
           f"{format_ts(elapsed_s, fmt)} s elapsed so far.")
    return f"{hdr}\n<video>\nHow many minutes of surgery remain?"


def render_answer_minutes(minutes: float, think: str | None = None) -> str:
    body = "" if think is None else f"<think>{think}</think>"
    return body + f"<answer>{float(minutes):.1f}</answer>"


def parse_minutes(text: str) -> float | None:
    text = text or ""
    m = _ANSWER_RE.search(text)
    inner = m.group(1) if m else text
    inner = re.sub(r"(?i)\b(minutes?|mins?)\b", "", inner)
    m2 = _MIN_RE.search(inner)
    return float(m2.group(0)) if m2 else None


# --------------------------------------------------------------------------- #
# grounded QA (T5), dense detection (T4), interval summary (T6)
# --------------------------------------------------------------------------- #
SYSTEM_QA = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds (total duration given). Answer the "
    "question about THIS video only. If it cannot be answered from the video, or assumes "
    "an impossible order of events, answer exactly <answer>ABSTAIN</answer>. Think briefly "
    "in <think>...</think>, then give a short <answer>...</answer>."
)
SYSTEM_DETECTION = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds. List every item active within the "
    "asked window, comma-separated, inside <answer>...</answer> (or <answer>none</answer>)."
)
SYSTEM_SUMMARY = (
    "You are a surgical video assistant. You see frames sampled across ONE complete "
    "operation, each with its timestamp in seconds. Summarize what happens in the asked "
    "time interval in 2-4 sentences, in chronological order, inside <answer>...</answer>."
)


def render_user_qa(frame_t_sec, duration_s: float, question: str, fmt: str = "seconds") -> str:
    return f"{render_frame_header(frame_t_sec, duration_s, fmt)}\n<video>\n{question}"


def render_user_detection(frame_t_sec, duration_s: float, t0: float, t1: float,
                          kind: str = "steps", fmt: str = "seconds") -> str:
    return (f"{render_frame_header(frame_t_sec, duration_s, fmt)}\n<video>\n"
            f"List the active {kind} in {format_ts(t0, fmt)}-{format_ts(t1, fmt)} s.")


def render_user_summary(frame_t_sec, duration_s: float, t0: float, t1: float,
                        fmt: str = "seconds") -> str:
    return (f"{render_frame_header(frame_t_sec, duration_s, fmt)}\n<video>\n"
            f"Summarize {format_ts(t0, fmt)}-{format_ts(t1, fmt)} s of this operation.")


def render_answer_labels(labels, think: str | None = None) -> str:
    body = "" if think is None else f"<think>{think}</think>"
    inner = ", ".join(str(x) for x in labels) if labels else "none"
    return body + f"<answer>{inner}</answer>"


def render_answer_text(text: str, think: str | None = None) -> str:
    body = "" if think is None else f"<think>{think}</think>"
    return body + f"<answer>{text.strip()}</answer>"


def parse_labels(text: str) -> list[str]:
    """Comma / semicolon separated label list from an <answer> block; [] for 'none'."""
    text = text or ""
    m = _ANSWER_RE.search(text)
    inner = (m.group(1) if m else text).strip()
    if not inner or re.fullmatch(r"(?i)none|n/?a|-", inner):
        return []
    return [p.strip() for p in re.split(r"[;,]", inner) if p.strip()]


def extract_answer_text(text: str) -> str:
    """Raw payload inside the first <answer>...</answer>, else the whole string trimmed."""
    m = _ANSWER_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()
