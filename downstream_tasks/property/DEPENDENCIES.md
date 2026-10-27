# Dependency versions and provenance

Versions below were read from the active Python environment used by the entry points of
this package. They are evidence of what was observed during preparation, not a complete lock of
every transitive package.

| Component | Observed version | Evidence |
| --- | --- | --- |
| Python | 3.11.0 | `python --version` |
| PyTorch | 2.5.0+cu124 | installed distribution metadata |
| transformers | 4.48.1 | installed distribution metadata |
| pandas | 2.2.3 | installed distribution metadata |
| NumPy | 2.4.4 | installed distribution metadata |
| scikit-learn | 1.6.1 | installed distribution metadata |
| SciPy | 1.15.2 | installed distribution metadata |
| PyYAML | 6.0.3 | installed distribution metadata |
| matplotlib | 3.10.0 | installed distribution metadata |
| tqdm | 4.67.3 | installed distribution metadata |

## Uni-RNA source and weights

The UniRNA implementation (`unirna_tf`) is the repository-level copy (CC BY-NC 4.0);
this package references it and does not bundle it.
`scripts/model.py` loads it from there by default; set `UR_UNIRNA_PATH` to use another
checkout.

The pretrained weights are **not** bundled; `UR_UNIRNA_WEIGHTS` gives the location of a local
Uni-RNA `unirna_L16` checkpoint directory (default convention: `weights/unirna_L16`).
Checkpoint acquisition terms, checksums, and access permissions are subject to the upstream
Uni-RNA license and author confirmation.

## Setup

```bash
python -m pip install -r requirements.txt
export UR_UNIRNA_WEIGHTS="$PWD/weights/unirna_L16"
```

The package itself needs no installation: `scripts/main.py` runs directly.
