# RNA-targeting small molecules screening pipeline

- `scripts/train.py` — training entry point: trains the bottleneck model on the shipped
  features and writes `results_repro/` (checkpoint + metrics).
- `scripts/infer.py` — inference entry point: rebuilds the model from a checkpoint and
  recomputes test-set predictions and metrics.
- `scripts/model.py` — encoder wrappers used by `extract_rna_features.py`.
- `scripts/extract_rna_features.py` and `scripts/extract_molecule_features.py` — the code
  that produced the feature files
- `configs/train.yaml` / `configs/test.yaml` — run configurations (parser defaults for
  `train.py` / `infer.py`; CLI arguments override).
- `data/` — train/test CSVs. Required to run: they carry the pairings and labels
  (id, rna_chain_sequences, ligand_smiles, target) that training and inference join
  against the shipped embeddings.
- `features/` — pre-extracted frozen embeddings: `Uni-RNA.pt` (RNA, keyed by sequence),
  `unimol_embeddings_dict.pt` (molecules, keyed by SMILES), stored in the hugging face link. 
- `third_party/` — vendored encoder code: unimol_lite (UniMol, MIT).

The standard output includes the best training weights `model.pt`, model metrics `metrics.json`, and prediction results `test_predictions.csv`.

## Environment

- Python 3.11 and the pinned direct dependencies in `requirements.txt`; see
  `DEPENDENCIES.md` for the provenance of the preparation environment.

## Run

From the package root:

```bash
# train
python scripts/train.py --config configs/train.yaml

# inference
python scripts/infer.py --config configs/test.yaml
```

## License

Third-party components and their terms are listed in `THIRD_PARTY_NOTICES.md`.
