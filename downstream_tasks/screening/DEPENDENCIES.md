# Dependencies

## Training / inference environment

Observed on the preparation machine (Linux x86-64, CUDA 12.4 driver, single GPU):

| Component | Version |
|-----------|---------|
| Python | 3.11.0 |
| PyTorch | 2.5.0+cu124 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| numpy | 2.4.4 |

These versions are the ones the results were produced and verified with
(last end-to-end verification: 2026-10-26). `requirements.txt` pins the direct
dependencies. To recreate the environment:

```bash
conda create -n ur_ligand python=3.11
conda activate ur_ligand
pip install -r requirements.txt
```

## Feature extraction extras

Only needed if you re-run the extraction scripts instead of using the shipped feature files.

| Component | Version |
|-----------|---------|
| transformers | 4.48.1 |
| rdkit | 2024.09.4 |
| scipy | (any recent) |

- Uni-RNA weights: overridable via `UR_UNIRNA_WEIGHTS` in `scripts/model.py`.
- UniMol weights: overridable via `UR_UNIMOL_WEIGHTS` in
  `scripts/extract_molecule_features.py`.
