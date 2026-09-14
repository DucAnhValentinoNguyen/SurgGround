"""QA metrics (PLAN.md 9.4): exact-match / token-F1, temporal-consistency rate,
and the plumbing for the P3 LLM-judge.

Pure functions (str/list in, float/dict out). The judge itself
(``Qwen2.5-7B-Instruct``, fixed {correct/partial/wrong} rubric) is wired in P3 —
``build_judge_prompt`` / ``aggregate_judge`` are the model-free halves and are
implemented here so P3 only adds the generation call.
"""
from __future__ import annotations

import re
import string


# --------------------------------------------------------------------------- #
# exact match / token F1  (SQuAD-style normalisation)
# --------------------------------------------------------------------------- #
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    n_same = 0
    g_left = list(g)
    for t in p:
        if t in g_left:
            g_left.remove(t)
            n_same += 1
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)


def _pairs(records):
    for r in records:
        yield (r.get("pred", r.get("pred_text", "")),
               r.get("gold", r.get("answer", r.get("gold_text", ""))))


def em_score(records) -> float:
    recs = list(records)
    return sum(exact_match(p, g) for p, g in _pairs(recs)) / len(recs) if recs else float("nan")


def f1_score(records) -> float:
    recs = list(records)
    return sum(token_f1(p, g) for p, g in _pairs(recs)) / len(recs) if recs else float("nan")


# --------------------------------------------------------------------------- #
# temporal-consistency vs the procedure graph
# --------------------------------------------------------------------------- #
def temporal_consistency_rate(records, graph) -> float:
    """Fraction of QA answers whose stated event ordering / step-phase pairing does
    **not** violate the procedure graph.

    A record may carry any of:
      * ``labeled_spans``   — [(label, t0, t1), ...] the answer commits to
      * ``ordered_labels``  — phase labels in the order the answer asserts them
      * ``step_label`` + ``phase_label`` — a step<->phase pairing
    Records with none of these are skipped.
    """
    if graph is None:
        return float("nan")
    checked = ok = 0
    for r in records:
        used = False
        if r.get("labeled_spans"):
            used = True
            ok += int(len(graph.violations(r["labeled_spans"])) == 0)
        elif r.get("ordered_labels"):
            used = True
            spans = [(lab, float(i), float(i) + 1.0) for i, lab in enumerate(r["ordered_labels"])]
            ok += int(len(graph.violations(spans)) == 0)
        elif r.get("step_label") is not None and r.get("phase_label") is not None:
            used = True
            ok += int(graph.step_phase_consistent(r["step_label"], r["phase_label"]))
        if used:
            checked += 1
    return ok / checked if checked else float("nan")


# --------------------------------------------------------------------------- #
# LLM-judge plumbing (body in P3)
# --------------------------------------------------------------------------- #
DEFAULT_RUBRIC = (
    "Grade the ANSWER against the REFERENCE for the QUESTION. Reply with exactly "
    "one word: correct (same meaning), partial (partially right or incomplete), or "
    "wrong (contradicts or misses the reference)."
)


def build_judge_prompt(record: dict, rubric: str = DEFAULT_RUBRIC) -> str:
    return (f"{rubric}\n\nQUESTION: {record.get('question', '').strip()}\n"
            f"REFERENCE: {str(record.get('gold', record.get('answer', ''))).strip()}\n"
            f"ANSWER: {str(record.get('pred', '')).strip()}\n\nVERDICT:")


_VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def parse_verdict(text: str) -> str:
    t = (text or "").strip().lower()
    for v in ("correct", "partial", "wrong"):
        if v in t:
            return v
    return "wrong"


def aggregate_judge(verdicts) -> dict:
    vs = [parse_verdict(v) for v in verdicts]
    if not vs:
        return {"qa_judge": float("nan"), "n": 0}
    return {
        "qa_judge": sum(_VERDICT_SCORE[v] for v in vs) / len(vs),
        "qa_judge_correct": vs.count("correct") / len(vs),
        "qa_judge_partial": vs.count("partial") / len(vs),
        "qa_judge_wrong": vs.count("wrong") / len(vs),
        "n": len(vs),
    }


def evaluate(records, graph=None) -> dict:
    """Model-free QA metric bundle. LLM-judge is added by ``judge`` in P3."""
    recs = list(records)
    return {
        "qa_em": em_score(recs),
        "qa_f1": f1_score(recs),
        "temporal_consistency": temporal_consistency_rate(recs, graph),
        "n": len(recs),
    }


def judge(records, cfg):
    raise NotImplementedError(
        "P3: run cfg.eval.judge_model over build_judge_prompt(...) then aggregate_judge(...)."
    )
