# qa_synth_cache/

Committed cache of offline QA paraphrases (PLAN.md 8.4, `surgground/data/qa_synth.py`).

Each `<dataset>_<split>.jsonl` line is `{"key", "orig", "paraphrase", "model", "seed"}`,
where `key = sha1(f"{model}\x1f{sub_type}\x1f{question}")[:16]`. `data.tasks.build()`
looks up the cache by key and substitutes the paraphrase into the T5 QA item's user
turn; a miss leaves the templated question verbatim — **no run ever calls an LLM**.

Regenerate (on a box with `cfg.data.qa_synth_model` available, e.g. `biostat`):

```
python -m surgground.data.qa_synth --build --dataset <ds> --split <split>
```

then commit the resulting `<ds>_<split>.jsonl`. This directory currently ships
empty (identity fallback only) — P2 wires the mechanism; the model is run later
once a QA item set from a real P1 dataset exists.
