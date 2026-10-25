# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.


"""Explicit loader for the RNAret RetNet backbone.

Instantiates ``RNAretBackbone`` at the hyperparameters used in the
upstream ``ssp.py`` fine-tuning script (the secondary-structure task we
share with upstream) and loads a pretrained ``pretrain_{k}mer.pth``
checkpoint from disk.

Upstream download
-----------------
    https://bis.zju.edu.cn/rnaret/download/model/   (model.tar.gz, ~937 MB)

After extracting, the k=1 weight file is::

    model/pretrain/pretrain_1mer.pth   (~47 MB)

This repository expects the extracted ``model/`` tree to live under
``<repo>/weights/rnaret/`` by default (see
``configs/pretrained/rnaret.yaml``).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .modeling_rnaret import RNAretBackbone
from .retnet import RetNetConfig


# Hyperparameters from upstream `ssp.py` (`parser` defaults for k=1).
# Source of truth: https://github.com/DrBlackZJU/RNAret/blob/main/ssp.py
RNARET_SSP_HPARAMS: dict = {
    "retnet_embed_dim": 384,
    "retnet_value_embed_dim": 512,
    "retnet_ffn_embed_dim": 512,
    "retnet_layers": 8,
    "retnet_retention_heads": 4,
    "dropout": 0.0,
    "activation_dropout": 0.0,
}


def _build_config(k: int, dropout: float, activation_dropout: float) -> RetNetConfig:
    return RetNetConfig(
        vocab_size=4**k + 6,
        retnet_embed_dim=RNARET_SSP_HPARAMS["retnet_embed_dim"],
        retnet_value_embed_dim=RNARET_SSP_HPARAMS["retnet_value_embed_dim"],
        retnet_ffn_embed_dim=RNARET_SSP_HPARAMS["retnet_ffn_embed_dim"],
        retnet_layers=RNARET_SSP_HPARAMS["retnet_layers"],
        retnet_retention_heads=RNARET_SSP_HPARAMS["retnet_retention_heads"],
        dropout=dropout,
        activation_dropout=activation_dropout,
    )


def load_rnaret_model(
    weights_path: str | Path,
    k: int = 1,
    dropout: float = 0.0,
    activation_dropout: float = 0.0,
) -> RNAretBackbone:
    """Build an ``RNAretBackbone`` at the SSP hyperparameters and load weights.

    Args:
        weights_path: Either a directory holding the upstream ``model/``
            tree (``<dir>/model/pretrain/pretrain_{k}mer.pth``), or the
            ``.pth`` file itself.
        k: k-mer size (matches upstream ``-k`` / ``k_num``).  **Default
            is 1**, i.e. single-nucleotide tokenization, which is
            upstream's recommended setting for SSP.  (Despite the
            project name, ``k`` is NOT a retrieval depth — RNAret has
            no retrieval mechanism; it is a RetNet-based language
            model.)
        dropout: Transformer dropout for the backbone.
        activation_dropout: Activation dropout for the backbone.

    Returns:
        ``RNAretBackbone`` with pretrained weights loaded.
    """
    weights_path = Path(weights_path)
    if weights_path.is_dir():
        candidate = weights_path / "model" / "pretrain" / f"pretrain_{k}mer.pth"
        if not candidate.exists():
            candidate = weights_path / f"pretrain_{k}mer.pth"
        weights_file = candidate
    else:
        weights_file = weights_path
    if not weights_file.exists():
        raise FileNotFoundError(
            f"RNAret pretrained checkpoint not found at {weights_file}. "
            f"Download model.tar.gz from "
            f"https://bis.zju.edu.cn/rnaret/download/model/ and extract so "
            f"that <weights_path>/model/pretrain/pretrain_{k}mer.pth exists."
        )

    config = _build_config(k, dropout=dropout, activation_dropout=activation_dropout)
    backbone = RNAretBackbone(config)

    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
    # Upstream ``ssp.py`` wraps the backbone in ``rnaret_ssp_model`` as
    # ``self.ret`` and loads ``pretrain_{k}mer.pth`` straight into the
    # RetNet submodule — so the keys already match our
    # ``self.retnet.*`` namespace.  Strict-load to catch any drift.
    backbone.retnet.load_state_dict(state_dict, strict=True)
    return backbone
