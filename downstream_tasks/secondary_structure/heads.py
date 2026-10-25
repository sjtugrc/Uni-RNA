#!/usr/bin/env python3
"""Custom RNA secondary-structure prediction heads for the MM-native runner.

Imported by ``train.py`` for its ``@HEADS.contact.logits.register`` side effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput

from multimolecule.modules.criterions import CRITERIONS
from multimolecule.modules.heads import HEADS, BasePredictionHead, HeadConfig, HeadOutput
from multimolecule.modules.networks import UNet


@HEADS.contact.logits.register("contact_head")
@HEADS.contact.logits.register("deeprnass_contact")
@HEADS.contact.logits.register("deeprnass")
class DeepRnaSSContactHead(BasePredictionHead):
    """DeepRNASS Q/K contact head using MM stripping, symmetry, and criterion."""

    output_name: str = "last_hidden_state"
    require_attentions: bool = False

    def __init__(self, config: Any, head_config: HeadConfig | None = None) -> None:
        super().__init__(config, head_config)
        hidden_size = int(self.config.hidden_size)
        self.qk_proj = nn.Linear(hidden_size, 2 * hidden_size)
        self.ffn = nn.Linear(1, hidden_size)
        self.decoder = nn.Linear(hidden_size, self.num_labels, bias=self.config.bias)
        self.criterion = CRITERIONS.build(self.config)

    def forward(
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **_: Any,
    ) -> HeadOutput:
        output = get_head_output(outputs, output_name or self.output_name)
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, attention_mask, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        q, k = self.qk_proj(output).chunk(2, dim=-1)
        contact_map = torch.matmul(q, k.transpose(-2, -1)).unsqueeze(-1)
        contact_map = contact_map + self.ffn(contact_map)
        logits = self.decoder(contact_map)
        logits = self.symmetrize(logits)
        logits = mask_contact_logits(logits, attention_mask)

        if labels is not None:
            if isinstance(labels, NestedTensor) and not isinstance(logits, NestedTensor):
                logits = labels.nested_like(logits, strict=False)
            return HeadOutput(logits, self.criterion(logits, labels))
        return HeadOutput(logits)


@HEADS.contact.logits.register("moe_head")
@HEADS.contact.logits.register("moe")
@HEADS.contact.logits.register("deeprnass_moe")
class DeepRnaSSMoEHead(BasePredictionHead):
    """DeepRNASS-style sequence-level MoE contact head."""

    output_name: str = "last_hidden_state"
    require_attentions: bool = False

    def __init__(self, config: Any, head_config: HeadConfig | None = None) -> None:
        super().__init__(config, head_config)
        hidden_size = int(self.config.hidden_size)
        num_experts = int(self.config.get("num_experts", 3))
        expert_hidden_size = int(
            self.config.get("expert_hidden_size", self.config.get("hidden_dim", self.config.get("moe_hidden_size", 64)))
        )
        self.min_loop_length = int(self.config.get("min_loop_length", 3))
        self.num_experts = num_experts

        self.qk_proj = nn.Linear(hidden_size, 2 * expert_hidden_size)
        self.expert_linears = nn.ModuleList(
            nn.Linear(expert_hidden_size, self.num_labels, bias=self.config.bias) for _ in range(num_experts)
        )
        self.expert_budgets = nn.ParameterList(
            nn.Parameter(torch.tensor(3.0 - i, dtype=torch.float32)) for i in range(num_experts)
        )
        self.router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expert_hidden_size),
            nn.GELU(),
            nn.Linear(expert_hidden_size, num_experts),
        )
        self.criterion = CRITERIONS.build(self.config)
        self.router_logits: Tensor | None = None

    def forward(
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **_: Any,
    ) -> HeadOutput:
        output = get_head_output(outputs, output_name or self.output_name)
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, attention_mask, _ = self.remove_special_tokens(output, attention_mask, input_ids)

        q, k = self.qk_proj(output).chunk(2, dim=-1)
        contact_features = torch.einsum("bic,bjc->bijc", q, k)
        contact_features = self.symmetrize(contact_features)

        token_mask = attention_mask.to(dtype=output.dtype).unsqueeze(-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)
        seq_embed = (output * token_mask).sum(dim=1) / denom
        router_logits = self.router(seq_embed)
        router_weights = torch.softmax(router_logits, dim=-1)
        self.router_logits = router_logits

        expert_outputs = []
        for i, expert in enumerate(self.expert_linears):
            budget = self.expert_budgets[i].clamp(0.1, 5.0)
            expert_outputs.append(budget * torch.tanh(expert(contact_features)))
        expert_stack = torch.stack(expert_outputs, dim=-1)
        logits = (expert_stack * router_weights.view(output.shape[0], 1, 1, 1, self.num_experts)).sum(dim=-1)
        logits = self.symmetrize(logits)
        logits = mask_short_loops(logits, self.min_loop_length)
        logits = mask_contact_logits(logits, attention_mask)

        if labels is not None:
            if isinstance(labels, NestedTensor) and not isinstance(logits, NestedTensor):
                logits = labels.nested_like(logits, strict=False)
            return HeadOutput(logits, self.criterion(logits, labels))
        return HeadOutput(logits)


@HEADS.contact.logits.register("pair_mlp_unet")
class PairMlpUNetContactHead(BasePredictionHead):
    """Pair-MLP features followed by MM UNet contact refinement."""

    output_name: str = "last_hidden_state"
    require_attentions: bool = False

    def __init__(self, config: Any, head_config: HeadConfig | None = None) -> None:
        super().__init__(config, head_config)
        hidden_size = int(self.config.hidden_size)
        pair_hidden_size = int(self.config.get("pair_hidden_size", min(hidden_size, 128)))
        distance_embedding_size = int(self.config.get("distance_embedding_size", pair_hidden_size))
        distance_boundaries = self.config.get("distance_boundaries", [3, 8, 16, 32, 64, 128, 256, 512])

        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, pair_hidden_size),
            nn.GELU(),
        )
        self.distance_embedding = nn.Embedding(len(distance_boundaries) + 1, distance_embedding_size)
        self.register_buffer("distance_boundaries", torch.tensor(distance_boundaries, dtype=torch.long), persistent=False)

        pair_input_size = 4 * pair_hidden_size + distance_embedding_size
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_size, pair_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(pair_hidden_size, pair_hidden_size),
            nn.GELU(),
        )
        self.decoder = UNet(
            num_layers=int(self.config.get("num_layers", 4)),
            hidden_size=pair_hidden_size,
            block=self.config.get("block", "auto"),
            num_channels=self.config.get("num_channels"),
            symmetric=self.config.get("symmetric", True),
            num_labels=self.num_labels,
        )
        self.criterion = CRITERIONS.build(self.config)

    def pair_features(self, token_features: NestedTensor) -> NestedTensor:
        """Build pair features as a NestedTensor, one element at a time at its own ``L_i`` (no
        padding), so the conv decoder runs per sequence and never sees padded positions.
        """
        boundaries = self.distance_boundaries.to(token_features.device)
        elements = []
        for element in token_features:  # element: dense [L_i, H]
            length = element.shape[0]
            left = element.unsqueeze(1).expand(length, length, -1)
            right = element.unsqueeze(0).expand(length, length, -1)
            distance = torch.bucketize(pair_distance(length, element.device), boundaries)
            distance_embedding = self.distance_embedding(distance)
            elements.append(torch.cat([left, right, left * right, (left - right).abs(), distance_embedding], dim=-1))
        return NestedTensor(elements)

    def forward(
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **_: Any,
    ) -> HeadOutput:
        output = get_head_output(outputs, output_name or self.output_name)
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, attention_mask, _ = self.remove_special_tokens(output, attention_mask, input_ids)
        if not isinstance(output, NestedTensor):
            output = NestedTensor.from_tensor_mask(output, attention_mask)

        token_features = self.token_proj(output)
        contact_map = self.pair_mlp(self.pair_features(token_features))
        contact_map = self.symmetrize(contact_map)
        logits = self.decoder(contact_map)
        logits = self.symmetrize(logits)
        logits = mask_contact_logits(logits, attention_mask)

        if labels is not None:
            if isinstance(labels, NestedTensor) and not isinstance(logits, NestedTensor):
                logits = labels.nested_like(logits, strict=False)
            return HeadOutput(logits, self.criterion(logits, labels))
        return HeadOutput(logits)


@HEADS.contact.logits.register("pair_mlp_stack_unet")
class PairMlpStackUNetContactHead(PairMlpUNetContactHead):
    """Pair-MLP + anti-diagonal stacking refinement + MM UNet contact head."""

    def __init__(self, config: Any, head_config: HeadConfig | None = None) -> None:
        super().__init__(config, head_config)
        pair_hidden_size = int(self.config.get("pair_hidden_size", min(int(self.config.hidden_size), 128)))
        self.stacking = StackingRefinement(
            hidden_size=pair_hidden_size,
            num_layers=int(self.config.get("stacking_layers", 2)),
            dropout=float(self.config.get("stacking_dropout", self.config.dropout)),
        )

    def forward(
        self,
        outputs: ModelOutput | Mapping[str, Tensor] | tuple[Tensor, ...],
        attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_name: str | None = None,
        **_: Any,
    ) -> HeadOutput:
        output = get_head_output(outputs, output_name or self.output_name)
        if attention_mask is None:
            attention_mask = self.get_attention_mask(input_ids)
        output, attention_mask, _ = self.remove_special_tokens(output, attention_mask, input_ids)
        if not isinstance(output, NestedTensor):
            output = NestedTensor.from_tensor_mask(output, attention_mask)

        token_features = self.token_proj(output)
        contact_map = self.pair_mlp(self.pair_features(token_features))
        contact_map = self.symmetrize(contact_map)
        # Anti-diagonal stacking runs per element on its own L_i x L_i map (no padding to shift across).
        contact_map = NestedTensor([self.stacking(element.unsqueeze(0)).squeeze(0) for element in contact_map])
        logits = self.decoder(contact_map)
        logits = self.symmetrize(logits)

        if labels is not None:
            if isinstance(labels, NestedTensor) and not isinstance(logits, NestedTensor):
                logits = labels.nested_like(logits, strict=False)
            return HeadOutput(logits, self.criterion(logits, labels))
        return HeadOutput(logits)


class StackingRefinement(nn.Module):
    """Residual refinement along RNA helix stacking directions in a contact map."""

    def __init__(self, hidden_size: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(StackingRefinementLayer(hidden_size, dropout=dropout) for _ in range(num_layers))

    def forward(self, pair_features: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        pair_features = mask_pair_features(pair_features, attention_mask)
        for layer in self.layers:
            pair_features = layer(pair_features)
            pair_features = (pair_features + pair_features.transpose(1, 2)) / 2
            pair_features = mask_pair_features(pair_features, attention_mask)
        return pair_features


class StackingRefinementLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.update = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

    def forward(self, pair_features: Tensor) -> Tensor:
        normalized = self.norm(pair_features)
        inner_stack = shift_pair_features(normalized, row_shift=1, col_shift=-1)
        outer_stack = shift_pair_features(normalized, row_shift=-1, col_shift=1)
        update = self.update(torch.cat([normalized, inner_stack, outer_stack], dim=-1))
        return pair_features + self.gate(normalized) * update


def get_head_output(outputs: ModelOutput | Mapping[str, Tensor] | tuple[Tensor, ...], output_name: str) -> Tensor:
    if isinstance(outputs, (Mapping, ModelOutput)):
        return outputs[output_name]
    if isinstance(outputs, tuple):
        return outputs[0]
    raise ValueError(f"Unsupported type for outputs: {type(outputs)}")


def pair_distance(length: int, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device)
    return (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()


def shift_pair_features(pair_features: Tensor, row_shift: int, col_shift: int) -> Tensor:
    length = pair_features.shape[1]
    shifted = pair_features.new_zeros(pair_features.shape)

    source_row_start = max(0, row_shift)
    source_row_end = length + min(0, row_shift)
    source_col_start = max(0, col_shift)
    source_col_end = length + min(0, col_shift)
    if source_row_start >= source_row_end or source_col_start >= source_col_end:
        return shifted

    target_row_start = max(0, -row_shift)
    target_col_start = max(0, -col_shift)
    target_row_end = target_row_start + source_row_end - source_row_start
    target_col_end = target_col_start + source_col_end - source_col_start
    shifted[:, target_row_start:target_row_end, target_col_start:target_col_end] = pair_features[
        :, source_row_start:source_row_end, source_col_start:source_col_end
    ]
    return shifted


def mask_pair_features(pair_features: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return pair_features
    pair_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    return pair_features.masked_fill(~pair_mask.bool().unsqueeze(-1), 0.0)


def mask_contact_logits(logits: Tensor | NestedTensor, attention_mask: Tensor | NestedTensor | None) -> Tensor | NestedTensor:
    if attention_mask is None:
        return logits
    if isinstance(logits, NestedTensor):
        if isinstance(attention_mask, NestedTensor):
            token_masks = list(attention_mask)
        else:
            token_masks = list(attention_mask)
        elements = []
        for index, element in enumerate(logits):
            token_mask = token_masks[index].reshape(-1)[: element.shape[0]].to(
                device=element.device, dtype=torch.bool
            )
            pair_mask = token_mask.unsqueeze(1) & token_mask.unsqueeze(0)
            if element.ndim > 2:
                pair_mask = pair_mask.unsqueeze(-1)
            elements.append(element.masked_fill(~pair_mask, 0.0))
        return NestedTensor(elements)
    pair_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    return logits.masked_fill(~pair_mask.bool().unsqueeze(-1), 0.0)


def mask_short_loops(logits: Tensor, min_loop_length: int) -> Tensor:
    if min_loop_length < 0:
        return logits
    length = logits.shape[1]
    idx = torch.arange(length, device=logits.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= min_loop_length
    return logits.masked_fill(mask.unsqueeze(0).unsqueeze(-1), -15.0)
