# Dependency versions and provenance

Versions below were read from the active Python environment used by the training entry points during package preparation. This records evidence, not a complete lock of every transitive package.

| Component | Observed version / pin | Evidence |
| --- | --- | --- |
| Python | 3.13.1 | `python --version` in the active training environment |
| PyTorch | 2.11.0+cu129 | installed distribution metadata |
| Transformers | 5.8.1 | installed distribution metadata |
| MultiMolecule | checkout `f46a10b1ff767630f20ae7c1f77f4d5ba4c9d00d` plus `third_party/patches/multimolecule-local.patch` | editable source checkout; metadata version was `0.1.1.dev9+g008a90f72` and is stale relative to checkout, so use the commit and patch |
| DanLing | `c5555bf568ae00c9830be21742a104d68795adac` | editable source checkout; tracked files clean; metadata version was `0.4.1.dev2+gd0e680101` and is stale relative to checkout |
| chanfig | 0.0.115 | installed distribution metadata |
| datasets | 4.8.5 | installed distribution metadata |
| pandas | 3.0.3 | installed distribution metadata |
| NumPy | 2.4.5 | installed distribution metadata |
| PyYAML | 6.0.3 | installed distribution metadata |
| PyArrow | 22.0.0 | installed distribution metadata; parquet reader |
| safetensors | 0.7.0 | installed distribution metadata |
| Biopython | 1.87 | installed distribution metadata |
| FairScale | 0.4.0 | installed distribution metadata; RNAret adapter |
| TorchScale | 0.3.0 | installed distribution metadata; RNAret adapter |

The nested UniRNA source has a clean local Git checkout at `d2a45cbc4ca0fd0d1affa4178c2f4d38bd899d2d`; only its runtime Python/tokenizer files and its existing license/copyright notices are included. The source project's large upstream `requirements.txt` is not copied because it is a notebook/environment dump rather than a minimal runtime lock and pins different Transformers/NumPy versions from the environment tested here.

## Install the public source dependencies

From the bundle root, with Git and network access:

```bash
mkdir -p external
git clone https://github.com/ZhiyuanChen/danling.git external/danling
git -C external/danling checkout c5555bf568ae00c9830be21742a104d68795adac
git clone https://github.com/DLS5-Omics/multimolecule.git external/multimolecule
git -C external/multimolecule checkout f46a10b1ff767630f20ae7c1f77f4d5ba4c9d00d
git -C external/multimolecule apply ../../third_party/patches/multimolecule-local.patch
python -m pip install -r requirements-runtime.txt
python -m pip install torch==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e external/danling
python -m pip install -e external/multimolecule
export PYTHONPATH="$PWD/third_party/unirna_code${PYTHONPATH:+:$PYTHONPATH}"
```

The pinned Torch wheel reflects the measured CUDA 12.9 environment. Select a compatible PyTorch build for another accelerator/runtime; that alternative was not tested here. MultiMolecule and DanLing are public GitHub repositories, but reproducibility requires the exact commits above; a generic latest PyPI install is not equivalent to this environment.
