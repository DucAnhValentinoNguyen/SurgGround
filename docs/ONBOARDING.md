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

### On **helena** — first prompt to the agent
> Read `CLAUDE.md` and run `bash scripts/agent_bootstrap.sh`. You are on the
> TRAINING box. First job: acquire the training datasets per `docs/DATASETS.md`
> — run `scripts/download/cholec80.sh`, `scripts/download/multibypass140.sh`,
> `scripts/download/grasp.sh`, then `autolaparo.sh` / `cholect50.sh` once I paste
> their URLs. Then implement **P1** (`surgground/data/*` parsers + `decode.py` +
> `splits.py`, finalize `procedure_graphs/{grasp,multibypass140}.json`) per
> `PLAN.md`. Update `docs/STATUS.md` as you go; branch `feat/p01-data`.

### On **biostat** — first prompt to the agent
> Read `CLAUDE.md` and run `bash scripts/agent_bootstrap.sh`. You are on the
> EVAL/DEV box. P0 scaffold is already in `main` (verify: `pytest -q` green).
> Start **P2** (`surgground/data/tasks.py` + `templates` wiring + `shards.py` +
> `collate.py` + the remaining `surgground/eval/*` metric modules — grounding /
> phase / rsd / reliability are already implemented with tests, do detection /
> qa / summary / efficiency / aggregate). Use `scripts/download/charades_sta.sh`
> as the stand-in. Also help finalize `feat/p01-parsers` code (no data needed).
> Branch `feat/p02-tasks-metrics`. Update `docs/STATUS.md`.

Both agents: `git pull` at session start, `git push` + update `docs/STATUS.md`
before stopping. The `STATUS.md` **Box** column is the lock — claim a phase row
before working it.
