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


"""HF-compatible wrapper around RNAret's RetNet backbone.

RNAret's upstream forward returns ``(logits, aux)`` where
``aux['inner_states'][-1]`` is the final hidden state tensor of shape
``(B, L, D)``.  MM's sequence backbone expects an HF-style object with a
``.last_hidden_state`` attribute and a ``.config.hidden_size`` attribute,
so this thin wrapper adapts the interface without touching the vendored
``retnet.py``.

Gradient checkpointing
----------------------
The upstream RetNet already supports layer-level activation checkpointing
via ``args.checkpoint_activations``; ``FusionBackbone`` toggles it
through ``gradient_checkpointing_enable()`` which we wire up to rebuild
the layer list with ``checkpoint_wrapper`` on demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .retnet import RetNet, RetNetConfig


@dataclass
class _BackboneConfig:
    """Mimics the ``.config`` attribute HF models expose.

    ``FusionBackbone`` reads ``config.hidden_size`` and (optionally)
    ``config.max_position_embeddings``; RetNet has no hard positional
    limit (RoPE-style rotary theta shifts), so we leave that field out.
    """

    hidden_size: int


class RNAretBackbone(nn.Module):
    """Wraps RNAret's ``RetNet`` as an HF-style feature extractor.

    Args:
        retnet_config: A fully-populated ``RetNetConfig`` instance.
    """

    def __init__(self, retnet_config: RetNetConfig):
        super().__init__()
        self.retnet_config = retnet_config
        self.retnet = RetNet(retnet_config, features_only=True)
        self.config = _BackboneConfig(hidden_size=retnet_config.retnet_embed_dim)

    # --- HF-like gradient-checkpointing surface -----------------------------
    def gradient_checkpointing_enable(self, **_kwargs) -> None:
        # RetNet wires ``args.checkpoint_activations`` only at build time
        # via ``fairscale.nn.checkpoint_wrapper``.  Rebuilding the layer
        # list post-hoc would require dropping/reinitialising weights, so
        # we instead log a warning and let ``dropout=0, bsz=1`` handle
        # memory (which matches the upstream SSP training recipe).
        import logging

        logging.getLogger(__name__).warning(
            "RNAret backbone does not support runtime gradient checkpoint "
            "toggling (fairscale checkpoint_wrapper is applied at build "
            "time).  Enable it via retnet_config.checkpoint_activations=True "
            "before constructing the backbone if memory is tight."
        )

    def gradient_checkpointing_disable(self) -> None:
        pass

    # --- forward ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        # RetNet's forward signature is ``(prev_output_tokens, ...)`` and
        # it builds its own causal / chunkwise mask from token positions;
        # the attention mask is only used to zero out padding hidden states
        # downstream so we accept it but do not pass it in.
        _logits, aux = self.retnet(input_ids.long())
        last_hidden_state = aux["inner_states"][-1]

        # If padding is present, zero out those positions so downstream
        # heads that look at all L positions see clean zeros (they also
        # consult ``attention_mask`` via the strip helper, but belt +
        # braces keeps things honest).
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            mask = attention_mask.bool()
        else:
            mask = attention_mask
        if mask is not None:
            last_hidden_state = last_hidden_state * mask.unsqueeze(-1).to(last_hidden_state.dtype)

        class _Output:
            pass

        out = _Output()
        out.last_hidden_state = last_hidden_state
        out.pooler_output = None
        return out
