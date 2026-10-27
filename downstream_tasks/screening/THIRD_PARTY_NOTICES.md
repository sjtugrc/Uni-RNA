# Third-party notices

This package does not assign a license to the top-level research code.

## Vendored code

- `third_party/unimol_lite/` — the UniMol molecule encoder (DP Technology), whose source
  headers carry the **MIT** license; `README.md` inside is the bundled upstream readme.

## Referenced but not bundled

- UniMol pretrained weights (`base_unimol`) — external pretrained asset; location
  overridable via `UR_UNIMOL_WEIGHTS` in `scripts/extract_molecule_features.py`.

## Scientific Python stack

PyTorch (BSD-style), scikit-learn (BSD-3), pandas (BSD-3), numpy (BSD-3),
transformers (Apache-2.0), RDKit (BSD-3), SciPy (BSD-3) — installed via pip from public
index mirrors.

## Datasets and derived artifacts

RNA and molecule feature files under `features/` are extracted embeddings derived from the
encoders used by this package; datasets under `data/` are experiment data of this study.

This package is not a blanket permission to redistribute third-party code, model weights,
or datasets.
