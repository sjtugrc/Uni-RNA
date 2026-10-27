# ur_bench_round2 — Uni-RNA Downstream Benchmark

- `scripts/main.py` — the entry point: CLI argument parsing, config merging, and mode
  dispatch (`train` / `post_train` / `infer`).
- `scripts/engine.py` — training/evaluation loops and metrics.
- `scripts/model.py` — `UniRNAEncoder` + `build_model`; Uni-RNA source/weight locations,
  overridable via `UR_UNIRNA_PATH` / `UR_UNIRNA_WEIGHTS`.
- `scripts/data.py` — csv/pkl loading and dataset construction.
- `scripts/utils.py` — seed, logging, and checkpoint helpers.
- `configs/train.yaml` / `configs/test.yaml` — run-configuration templates; a YAML config
  supplies parser defaults, CLI arguments override it. `--patience` is honored when set in
  either place.
- Run outputs (checkpoints, logs, predictions) go to the `--output_dir` given on the
  command line.
- The UniRNA implementation (`unirna_tf`) is the repository-level copy; this package
  references it and does not bundle it.

## Environment

- Use Python 3.11 and the direct dependency versions in `requirements.txt` to match the
  environment observed during preparation; see `DEPENDENCIES.md` for provenance details.
- The UniRNA implementation (`unirna_tf`) is the repository-level copy (used by default;
  override via `UR_UNIRNA_PATH`). Pretrained weights are not bundled; the weights directory
  defaults to `weights/unirna_L16`, overridable via `UR_UNIRNA_WEIGHTS`.

The package itself needs no installation — run the script directly.

## Run

```bash
# train
python scripts/main.py --mode train --config configs/train.yaml \
  --output_dir outputs/<run> --gpu 0

# inference
python scripts/main.py --mode infer --config configs/test.yaml \
  --saved_model_path <checkpoint> --output_dir outputs/<run> --gpu 0
```

## Notes

- The learning rate comes from `base_lr` in the config file; CLI arguments override config
  values.
- Early stopping is off by default (`--patience 0`); set `patience` in the config or pass
  `--patience N` to enable it.
- Metrics are parsed from `[Eval]` / `[Infer]` log lines; `[Train]` lines are training-set
  scores.
- The GPU is selected via `--gpu`.

## License

The UniRNA implementation (`unirna_tf`) is licensed CC BY-NC 4.0; this package references
the repository-level copy and does not bundle it. Pretrained weights and
datasets are referenced by configs/environment and are not part of this repository.
Dataset and checkpoint URLs, checksums, access permissions, and data-use terms need
author confirmation.
