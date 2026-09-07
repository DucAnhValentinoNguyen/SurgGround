"""Procedure graph: the phase/step partial order (PLAN.md H2, Appendix A). REAL (P0).

Three uses:
  1. decode-time logit mask when the model emits a phase/step LIST (`forbidden_next`)
  2. the `r_order` reward term in RL (`violations`)   -- see surgground/train/rewards.py
  3. an eval metric: order-violation rate / step<->phase consistency (P6, P9)

JSON schema (procedure_graphs/<dataset>.json):
  { "dataset", "phases": [{"id","name"}], "steps": [{"id","name","parent_phase"}]?,
    "hard_precede": [[a,b],...], "soft_precede": [[a,b],...],
    "mutually_exclusive_at_t": bool }
`hard_precede: [a, b]` means "phase a MUST fully precede phase b".
"""
from __future__ import annotations

import json
from pathlib import Path


class ProcedureGraph:
    def __init__(self, dataset, phases, steps, hard_precede, soft_precede,
                 mutually_exclusive_at_t=True):
        self.dataset = dataset
        self.phases = list(phases)                       # [{id, name}]
        self.steps = list(steps or [])                   # [{id, name, parent_phase}]
        self.hard = [tuple(p) for p in (hard_precede or [])]
        self.soft = [tuple(p) for p in (soft_precede or [])]
        self.mutually_exclusive_at_t = bool(mutually_exclusive_at_t)

        self._name2id = {p["name"]: p["id"] for p in self.phases}
        self._name2id_ci = {p["name"].lower(): p["id"] for p in self.phases}
        self._id2name = {p["id"]: p["name"] for p in self.phases}
        self._step_parent = {s["id"]: s.get("parent_phase") for s in self.steps}
        self._hard_closure = _transitive_closure(self.hard)

    # ----- construction ----------------------------------------------------- #
    @classmethod
    def from_json(cls, path: str | Path) -> "ProcedureGraph":
        d = json.loads(Path(path).read_text())
        return cls(
            dataset=d.get("dataset", Path(path).stem),
            phases=d["phases"],
            steps=d.get("steps", []),
            hard_precede=d.get("hard_precede", []),
            soft_precede=d.get("soft_precede", []),
            mutually_exclusive_at_t=d.get("mutually_exclusive_at_t", True),
        )

    # ----- queries -------------------------------------------------------- #
    def phase_id(self, label) -> int | None:
        if isinstance(label, int):
            return label
        return self._name2id.get(label) or self._name2id_ci.get(str(label).lower())

    def phase_name(self, pid: int) -> str:
        return self._id2name.get(pid, str(pid))

    def step_parent(self, step_id: int) -> int | None:
        return self._step_parent.get(step_id)

    def must_precede(self, a, b) -> bool:
        """True if phase a must fully precede phase b (transitively)."""
        a, b = self.phase_id(a), self.phase_id(b)
        return a is not None and b is not None and b in self._hard_closure.get(a, set())

    def forbidden_next(self, emitted_ids, candidate_ids):
        """Given phases already emitted (in order), the candidate ids that would
        violate a hard precedence if emitted next: any c such that c must precede
        some already-emitted phase."""
        emitted = {self.phase_id(e) for e in emitted_ids}
        out = []
        for c in candidate_ids:
            cid = self.phase_id(c)
            if cid is None:
                continue
            if any(self.must_precede(cid, e) for e in emitted):
                out.append(c)
        return out

    def violations(self, labeled_spans):
        """`labeled_spans`: list of (label, t_start, t_end). Returns the list of
        (earlier_label, later_label) pairs whose ordering breaks a hard precedence
        or a step<->phase containment rule."""
        items = []
        for lab, t0, t1 in labeled_spans:
            items.append((lab, float(t0), float(t1)))
        items.sort(key=lambda x: x[1])
        bad = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                li, lj = items[i][0], items[j][0]
                # i starts no later than j; a violation is: j must precede i
                if self.must_precede(lj, li):
                    bad.append((items[i][0], items[j][0]))
        return bad

    def step_phase_consistent(self, step_label, phase_label) -> bool:
        sid = step_label if isinstance(step_label, int) else \
            next((s["id"] for s in self.steps if s["name"] == step_label), None)
        parent = self._step_parent.get(sid) if sid is not None else None
        return parent is None or parent == self.phase_id(phase_label)


def _transitive_closure(edges):
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    closure: dict[int, set[int]] = {}
    for src in list(adj):
        seen, stack = set(), [src]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        closure[src] = seen
    return closure
