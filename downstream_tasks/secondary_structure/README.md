## Secondary structure section

- `train.py`: MultiMolecule-backed fine-tuning runner; training is followed by structure inference/export.
- `infer.py`: inference/export used by `train.py`; reconstructs the network from the run's saved `train/metadata/config.full.yaml` and strictly loads the checkpoint.

- `modules.py`, `heads.py`, `aido_rna_impl/`, `rnaret_impl/`: local encoder/head adapters and source implementations needed by configured model choices.
- `configs/`: the train/inference settings. Select a model with `--model`.
- `requirements.txt`, `DEPENDENCIES.md`, `THIRD_PARTY_NOTICES.md`, `third_party/patches/`, `third_party/licenses/`: tested version evidence, exact source pins, preserved notices, and the small local MultiMolecule patch.

## Environment setup

Use Python 3.13.1 and the direct dependency versions in `requirements.txt` to match the environment observed during preparation. Install the pinned DanLing and MultiMolecule source revisions and apply the included patch by following `DEPENDENCIES.md`. Set the bundled UniRNA implementation on `PYTHONPATH`:

```bash
export PYTHONPATH="path/to/unirna_tf${PYTHONPATH:+:$PYTHONPATH}"
```

The observed PyTorch build was `2.11.0+cu129`. The corresponding install command is in `DEPENDENCIES.md`; a CPU or other CUDA build was not smoke-tested here.

## Run the configured experiments

Each command starts training and then runs configured inference; run them only after installing dependencies and staging the external assets above. Defaults set seed `1016`, and output paths resolve under `outputs/<task>/<experiment>/<model>`.

```bash
python train.py --config configs/unirna/unirna.yaml --model unirna

python train.py --config configs/bprna/bprna.yaml --model unirna

python train.py --config configs/archiveii/archiveii.yaml --model unirna
```

The effective model is saved with the run metadata; `infer.py` loads that saved network shape and fails clearly if the checkpoint output dimension disagrees. 

## License Information

- The CC BY-NC 4.0 UniRNA source license and the AGPL-3.0 MultiMolecule dependency require author/license review before sending or redistributing the package.
- Dataset and checkpoint URLs, checksums, access permissions, and data-use terms still need author confirmation.
