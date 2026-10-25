# Third-party notices and license status

This bundle does not assign a license to the top-level research code. No top-level project `LICENSE` was present in the audited source, and none has been added.


- `aido_rna_impl/` retains its source-file copyright/license headers. Those files identify Apache-2.0-licensed upstream code; the full Apache-2.0 text is included at `third_party/licenses/Apache-2.0.txt`.
- `rnaret_impl/` retains the MultiMolecule AGPL-3.0 notices in its adapter/tokenizer files. `third_party/licenses/AGPL-3.0-MultiMolecule.txt` contains the upstream package's full license text. `rnaret_impl/retnet.py` states that it is Microsoft TorchScale code under MIT; the full MIT text is included at `third_party/licenses/TorchScale-MIT.txt`.
- `third_party/patches/multimolecule-local.patch` modifies three files from the pinned MultiMolecule AGPL-3.0 source. The patch is included to reproduce the current runtime behavior, and the relevant license text is included above.
- DanLing is installed from the pinned public GitHub source commit, not copied into this archive. The installed source metadata offered the license expression `Unlicense OR AGPL-3.0-or-later OR GPL-3.0-or-later OR GPL-2.0-or-later OR Apache-2.0 OR MIT OR BSD-4-Clause`; consult the upstream repository for the selected terms.

The archive is not a blanket permission to redistribute third-party code, model weights, or datasets. No credentials, private keys, model weights, datasets, checkpoints, outputs, caches, or Git metadata are included.
