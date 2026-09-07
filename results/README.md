# results/

Git-ignored except this file. `surgground-eval` writes
`results/<backend>__<method>__<dataset>__<task>__<split>.json` here;
`surgground-aggregate` produces `long_results.csv` + markdown pivots.
Nothing in here is committed — checkpoints go to the HF Hub repo
(`scripts/sync_checkpoints.sh`), not git.
