# SurgGround — machine bring-up (helena + biostat)

Run once per box. ~15 min each. Then hand the box to an agent with the prompt at
the bottom.

---

## 0. Roles (ADR-014)

| Box | Role | Owns phases |
|---|---|---|
| **helena** (NVMe root, ~308 GB) | **training critical path** | P1 decode, P4 SFT, P5 sweep+retriever, P7 train, P10 |
| **biostat** (HDD root, ~251 GB; NAS `ra92miz`) | **eval / dev / baselines / ablations / demo** | P0 verify, P2, P3, P5 regime-eval, P6, P7 rollouts+eval, P8, P9, P11 |

No cross-box distributed training. Code syncs via git; checkpoints via
`scripts/sync_checkpoints.sh` (HF Hub); frame subsets via `rsync -e ssh`.

---

## 1. Clone + venv (both boxes)

```bash
git clone https://github.com/DucAnhValentinoNguyen/SurgGround.git
cd SurgGround
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv missing
bash scripts/setup_env_4090.sh                            # ~10 min; prints a capability report
```

Expect the report to show `NVIDIA GeForce RTX 4090  24564 MiB`, `bnb 4-bit OK`,
and `surgground 0.1.0 importable`. `flash-attn` may say ABSENT — fine (sdpa
fallback).

## 2. Per-box paths + env

```bash
cp scripts/env_4090.local.sh.example scripts/env_4090.local.sh
$EDITOR scripts/env_4090.local.sh        # set DATA_ROOT etc. for THIS box (examples in the file)
source scripts/env_4090.sh               # prints: box=helena|biostat  DATA_ROOT=...  ...
```
Defaults if you skip the edit: helena `DATA_ROOT=/data/surgground` + big files on
`/home`; biostat everything under `~/surgground`.

## 3. HF token (both boxes — helena pushes ckpts, biostat pulls)

```bash
printf '%s' 'hf_YOUR_WRITE_TOKEN' > ~/.hf_token && chmod 600 ~/.hf_token
huggingface-cli login --token "$(cat ~/.hf_token)"
huggingface-cli whoami                    # -> DucAnhValentinoNguyen
```
One-time (from any box): create the private repo —
`huggingface-cli repo create surgground-ckpts --type model --private`.

## 4. SSH between the boxes (for rsync)

See `docs/STATUS.md` B2b / the chat guide. Test:
`ssh -o BatchMode=yes biostat hostname` and the reverse must both work.
Add `Host helena` / `Host biostat` blocks to `~/.ssh/config` on each.

## 5. Verify P0

```bash
bash scripts/agent_bootstrap.sh          # branch, STATUS.md, capability
python -c "import surgground; print(surgground.__version__)"
python -m surgground.eval.run_eval --help
pytest -q                                 # pure modules green; later-phase tests skipped
ruff check .
```

## 6. System packages for the download scripts

```bash
sudo apt-get install -y aria2 unzip git ffmpeg
# gdown + synapseclient already installed by setup_env (`.[download]` extra)
```

---

## 7. Start the agents

**Order:** start the **biostat** agent first (P2 is fully unblocked and P3 depends
on its output), then the **helena** agent (kick off downloads, code parsers while
they run). The two branches touch disjoint files — they never block each other.

### On **biostat** — first prompt to the agent  (start this one first)
> Read `CLAUDE.md`, run `bash scripts/agent_bootstrap.sh`, and read
> `docs/STATUS.md` top to bottom. You are on the **biostat** EVAL/DEV box. P0 is
> in `main` — confirm `pytest -q` green. Claim the **P2** row in `docs/STATUS.md`
> (Box = biostat), branch `feat/p02-tasks-metrics`. Implement **P2** per
> `PLAN.md`: `surgground/data/{tasks,shards,collate,qa_synth}.py` and
> `surgground/eval/{detection,qa,summary,efficiency,aggregate}.py`.
> `eval/{grounding,phase,rsd,reliability}` + `data/{templates,regime}` are
> already implemented with tests — build on them, don't rewrite. Stand-in data:
> `scripts/download/charades_sta.sh`. **Do not touch**
> `surgground/data/{grasp,multibypass140,cholec80,cholect50,autolaparo,heichole}.py`
> or `procedure_graphs/*` (helena's, on `feat/p01-data`) — code `tasks.py`
> against the `class Parser` stub signatures already in the tree. Follow the DoD
> + smoke gate before any PR to `main`. Update `docs/STATUS.md` before stopping.

### On **helena** — first prompt to the agent
> Read `CLAUDE.md`, run `bash scripts/agent_bootstrap.sh`, and read
> `docs/STATUS.md` top to bottom. You are on the **helena** TRAINING box. Claim
> the **P1** row in `docs/STATUS.md` (Box = helena), branch `feat/p01-data`.
> **First**, kick off the ungated downloads detached:
> `nohup bash scripts/download/cholec80.sh > /tmp/dl_cholec80.log 2>&1 &` and the
> same for `multibypass140.sh` and `grasp.sh` (per `docs/DATASETS.md`);
> `autolaparo.sh` / `cholect50.sh` wait for URLs I paste. **While they run**,
> implement **P1** per `PLAN.md`: the six `surgground/data/*.py` dataset parsers,
> `decode.py`, `splits.py`, and the `_todo` fields in
> `procedure_graphs/{grasp,multibypass140}.json`. Follow the DoD + smoke gate
> before any PR to `main`. Record the download PIDs + log paths and update
> `docs/STATUS.md` before stopping.

Both agents: `git pull` at session start, `git push` + update `docs/STATUS.md`
before stopping. The `STATUS.md` **Box** column is the lock — claim a phase row
before working it.
