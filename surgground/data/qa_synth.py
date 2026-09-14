"""Offline QA paraphrase pass (PLAN.md 8.4, P2).

Rewrites *only the phrasing* of templated QA questions for fluency, using a LOCAL
instruct model (``cfg.data.qa_synth_model``), greedy + seeded. It **never**
touches a factual payload — ``target`` spans / labels / counts / minutes are
copied through untouched.

Determinism contract (CLAUDE.md "Determinism"): the paraphrases are produced
**once**, offline, by ``python -m surgground.data.qa_synth --build`` and committed
under ``surgground/data/qa_synth_cache/``. At task-build / train / eval time
``synth()`` only *looks up* the committed cache; a cache miss falls back to the
original question verbatim, so a run never calls an LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent / "qa_synth_cache"


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def _cache_key(question: str, sub_type: str, model: str) -> str:
    h = hashlib.sha1(f"{model}\x1f{sub_type}\x1f{question}".encode()).hexdigest()
    return h[:16]


def _load_cache(cache_dir: str | Path) -> dict[str, str]:
    cache_dir = Path(cache_dir)
    out: dict[str, str] = {}
    if not cache_dir.exists():
        return out
    for p in sorted(cache_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["key"]] = rec["paraphrase"]
    return out


def _user_question(item: dict) -> str | None:
    for m in item.get("messages", []):
        if m.get("role") == "user":
            return m["content"]
    return None


def _set_user_question(item: dict, text: str) -> None:
    for m in item.get("messages", []):
        if m.get("role") == "user":
            m["content"] = text
            return


# --------------------------------------------------------------------------- #
# public
# --------------------------------------------------------------------------- #
def synth(items, cfg=None, cache_dir: str | Path = _CACHE_DIR):
    """Apply committed paraphrases to T5 QA items in place; return ``items``.

    A cache miss leaves the question unchanged (no LLM call). Non-T5 items and
    every ``target`` are untouched.
    """
    model = "template"
    if cfg is not None:
        try:
            model = str(cfg.data.qa_synth_model)
        except Exception:
            pass
    cache = _load_cache(cache_dir)
    if not cache:
        return items
    for it in items:
        if it.get("task") != "t5":
            continue
        q = _user_question(it)
        if q is None:
            continue
        key = _cache_key(q, it.get("sub_type", ""), model)
        para = cache.get(key)
        if para and para.strip():
            _set_user_question(it, para.strip())
            it.setdefault("qa_synth", {})["key"] = key
    return items


def _paraphrase_with_model(questions: list[str], model_name: str, seed: int = 0) -> list[str]:
    """Greedy, seeded paraphrase with a local instruct model. Import-guarded so the
    package has no hard dependency on a loaded LLM; only ``--build`` calls this."""
    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    out = []
    sys = ("Rewrite the user's question so it sounds natural. Keep every name, "
           "number, time and label EXACTLY. Return only the rewritten question.")
    for q in questions:
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(mdl.device)
        gen = mdl.generate(ids, max_new_tokens=64, do_sample=False, temperature=None, top_p=None)
        out.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return out


def build_cache(dataset: str, split: str, cfg=None, cache_dir: str | Path = _CACHE_DIR,
                seed: int = 0) -> Path:
    """Offline, run on biostat with the judge/synth model present. Generates the
    committed ``<dataset>_<split>.jsonl`` paraphrase cache."""
    from .tasks import build

    cfg = cfg
    if cfg is None:
        from ..cfg import load_cfg
        cfg = load_cfg()
    model = str(getattr(getattr(cfg, "data", object()), "qa_synth_model", "template"))
    items = build(dataset, split, cfg, seed=seed, paraphrase=False, tasks=("t5",))
    seen: dict[str, str] = {}
    qs = []
    for it in items:
        if it.get("task") != "t5":
            continue
        q = _user_question(it)
        key = _cache_key(q, it.get("sub_type", ""), model)
        if key not in seen:
            seen[key] = q
            qs.append((key, q))
    paras = _paraphrase_with_model([q for _, q in qs], model, seed=seed)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{dataset}_{split}.jsonl"
    with out.open("w") as f:
        for (key, q), para in zip(qs, paras, strict=True):
            f.write(json.dumps({"key": key, "orig": q, "paraphrase": para,
                                "model": model, "seed": seed}) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="surgground.data.qa_synth", description=__doc__)
    p.add_argument("--build", action="store_true", help="generate + commit the paraphrase cache")
    p.add_argument("--dataset", default="standin")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.build:
        cache = _load_cache(_CACHE_DIR)
        print(f"[qa_synth] committed cache entries: {len(cache)}  ({_CACHE_DIR})")
        return 0
    out = build_cache(args.dataset, args.split, seed=args.seed)
    print(f"[qa_synth] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
