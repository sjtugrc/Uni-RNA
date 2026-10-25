#!/usr/bin/env python3
"""Encoder adapters and feature wrappers for the MM-native runner.

Imported by ``train.py`` for its ``@SEQUENCES.register`` side effects.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch
from danling import NestedTensor
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from multimolecule.modules.backbones.sequences import SEQUENCES
from multimolecule.tokenisers import RnaTokenizer


@dataclass
class _SequenceConfig:
    hidden_size: int
    pad_token_id: int = 0
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    num_labels: int = 1
    problem_type: str = "binary"


class _FeatureOutput(dict):
    """Mapping/object hybrid expected by MM's SequenceBackbone."""

    def __init__(self, last_hidden_state: Tensor, pooler_output: Tensor | None = None) -> None:
        if pooler_output is None:
            pooler_output = last_hidden_state[:, 0]
        super().__init__(last_hidden_state=last_hidden_state, pooler_output=pooler_output)
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


def build_onehot_tokenizer(**kwargs: Any) -> RnaTokenizer:
    """Build the exact nucleotide tokenizer used by the fixed one-hot adapter."""
    kwargs.setdefault("unk_token", "N")
    kwargs.setdefault("pad_token", "<pad>")
    kwargs.setdefault("mask_token", None)
    kwargs.setdefault("null_token", None)
    return RnaTokenizer(
        alphabet="streamline",
        bos_token=None,
        cls_token=None,
        eos_token=None,
        sep_token=None,
        **kwargs,
    )


@SEQUENCES.register("rna_onehot")
class RnaOneHotSequence(nn.Module):
    """Fixed A/C/G/U/N one-hot features with all padding positions set to zero."""

    supports_nested_tensor: bool = True
    BASES = ("A", "C", "G", "U", "N")

    def __init__(
        self,
        name: str | None = None,
        pretrained_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        tokenizer = build_onehot_tokenizer()
        vocab = tokenizer.get_vocab()
        token_ids = [int(tokenizer.convert_tokens_to_ids(base)) for base in self.BASES]
        if len(set(token_ids)) != len(self.BASES):
            raise ValueError("One-hot tokenizer must assign distinct ids to A, C, G, U, and N.")
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("One-hot tokenizer must define a padding token.")
        vocab_size = max(int(value) for value in vocab.values()) + 1
        id_to_channel = torch.full((vocab_size,), -1, dtype=torch.long)
        id_to_channel[token_ids] = torch.arange(len(self.BASES), dtype=torch.long)
        self.register_buffer("id_to_channel", id_to_channel, persistent=False)
        self.config = _SequenceConfig(hidden_size=len(self.BASES), pad_token_id=int(pad_token_id))

    def _encode(self, input_ids: Tensor) -> Tensor:
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.id_to_channel.numel()):
            raise ValueError("One-hot input contains a token id outside the RNA tokenizer vocabulary.")
        channels = self.id_to_channel[input_ids.long()]
        invalid = channels.lt(0) & input_ids.ne(self.config.pad_token_id)
        if invalid.any():
            raise ValueError("One-hot input contains a special token other than padding.")
        encoded = F.one_hot(channels.clamp_min(0), num_classes=len(self.BASES)).to(dtype=torch.float32)
        return encoded * channels.ge(0).unsqueeze(-1)

    def forward(
        self,
        input_ids: NestedTensor | Tensor,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> _FeatureOutput:
        if isinstance(input_ids, NestedTensor):
            if attention_mask is None:
                attention_mask = input_ids.mask
            input_ids = input_ids.tensor
            input_ids = input_ids.masked_fill(~attention_mask.bool(), self.config.pad_token_id)

        hidden = self._encode(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        hidden = hidden * attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return _FeatureOutput(hidden, hidden.sum(dim=-2))


@SEQUENCES.register("rnaret")
class RnaRetSequence(nn.Module):
    """RNAret RetNet adapter registered locally for this script."""

    def __init__(
        self,
        name: str | None = None,
        pretrained_path: str | None = None,
        k: int = 1,
        dropout: float = 0.0,
        activation_dropout: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        weights = pretrained_path or name
        if weights is None:
            raise ValueError("RnaRetSequence requires `pretrained_path` pointing to RNAret weights.")
        try:
            from rnaret_impl.loader import load_rnaret_model
        except ImportError as err:
            raise ImportError(
                "RNAret adapter requires optional dependencies `torchscale` and `fairscale`. "
                "Install them before using `encoder: rnaret`."
            ) from err
        self.module = load_rnaret_model(weights, k=k, dropout=dropout, activation_dropout=activation_dropout)
        self.config = self.module.config
        if not hasattr(self.config, "pad_token_id"):
            self.config.pad_token_id = 0

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.module, "gradient_checkpointing_enable"):
            self.module.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **_: Any) -> _FeatureOutput:
        output = self.module(input_ids, attention_mask=attention_mask)
        return _FeatureOutput(output.last_hidden_state, getattr(output, "pooler_output", None))


@SEQUENCES.register("aido_rna")
class AidoRnaSequence(nn.Module):
    """genbio AIDO.RNA adapter (RNABert: SwiGLU MLP + RoPE).

    genbio ships AIDO.RNA as weights only under ``model_type: "rnabert"``, which collides
    with multimolecule's RnaBert; we load the vendored genbio classes directly (via
    ``encoder: aido_rna``) so ``AutoModel`` never gets to mis-resolve the architecture.
    """

    def __init__(
        self,
        name: str | None = None,
        pretrained_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        weights = pretrained_path or name
        if weights is None:
            raise ValueError("AidoRnaSequence requires `pretrained` pointing to AIDO.RNA weights.")
        for consumed in ("use_pretrained", "gradient_checkpoint", "factory", "hidden_size", "k"):
            kwargs.pop(consumed, None)
        from aido_rna_impl import load_aido_rna_model

        self.module = load_aido_rna_model(weights, **kwargs)
        self.config = self.module.config
        if getattr(self.config, "pad_token_id", None) is None:
            self.config.pad_token_id = 0
        # AIDO.RNA's tokenizer wraps each sequence as ``[CLS] ... [SEP]`` (ids 2 and 3 in
        # vocab.txt). Expose them as bos/eos so the contact head strips both ends and the
        # (L, L) map aligns with the raw sequence (mirrors the UniRNA handling below).
        if getattr(self.config, "bos_token_id", None) is None:
            self.config.bos_token_id = 2  # [CLS]
        if getattr(self.config, "eos_token_id", None) is None:
            self.config.eos_token_id = 3  # [SEP]

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.module, "gradient_checkpointing_enable"):
            self.module.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **_: Any) -> _FeatureOutput:
        output = self.module(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return _FeatureOutput(output.last_hidden_state, getattr(output, "pooler_output", None))


@SEQUENCES.register("unirna")
class UniRnaSequence(nn.Module):
    """UniRNA adapter registered locally for this script."""

    def __init__(
        self,
        name: str | None = None,
        pretrained_path: str | None = None,
        factory: str | None = None,
        hidden_size: int | None = None,
        random_init: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        kwargs.pop("use_pretrained", None)
        kwargs.pop("gradient_checkpoint", None)
        self.module = self._load_module(
            pretrained_path or name, factory=factory, random_init=random_init, **kwargs
        )
        resolved_hidden = getattr(getattr(self.module, "config", None), "hidden_size", None) or hidden_size
        if resolved_hidden is None:
            raise ValueError("UniRNA adapter could not infer hidden_size. Pass `encoder_hidden_size`.")
        self.config = getattr(self.module, "config", _SequenceConfig(hidden_size=int(resolved_hidden)))
        self.config.hidden_size = int(resolved_hidden)
        if not hasattr(self.config, "pad_token_id"):
            self.config.pad_token_id = 0
        # UniRNA keeps CLS/EOS in the model input (matching pretraining); the MM head strips
        # them. Expose the special-token ids the head reads (UniRNAConfig names EOS `sep_token_id`
        # and leaves `bos_token_id`/`eos_token_id` as None), so both ends are stripped to align
        # with the (L, L) contact map.
        if getattr(self.config, "bos_token_id", None) is None:
            self.config.bos_token_id = getattr(self.config, "cls_token_id", None)
        if getattr(self.config, "eos_token_id", None) is None:
            self.config.eos_token_id = getattr(self.config, "sep_token_id", None)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.module, "gradient_checkpointing_enable"):
            self.module.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs: Any) -> _FeatureOutput:
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("output_attentions", False)
        kwargs.setdefault("return_dict", True)
        try:
            output = self.module(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        except TypeError:
            output = self.module(input_ids, attention_mask=attention_mask, **kwargs)

        if isinstance(output, Tensor):
            hidden = output
            pooler = None
        elif isinstance(output, dict):
            hidden = output.get("last_hidden_state")
            if hidden is None:
                hidden = output.get("logits")
            pooler = output.get("pooler_output")
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = getattr(output, "logits", None)
            pooler = getattr(output, "pooler_output", None)
        if hidden is None:
            raise ValueError("UniRNA module output must contain `last_hidden_state`, `logits`, or be a Tensor.")
        return _FeatureOutput(hidden, pooler)

    @staticmethod
    def _load_module(
        source: str | None, *, factory: str | None, random_init: bool = False, **kwargs: Any
    ) -> nn.Module:
        if random_init and factory is not None:
            raise ValueError("Random-init UniRNA requires the registered local model classes, not encoder_factory.")
        if factory is not None:
            module_name, attr = factory.replace(":", ".").rsplit(".", 1)
            loader = getattr(importlib.import_module(module_name), attr)
            return loader(source, **kwargs) if source is not None else loader(**kwargs)

        try:
            importlib.import_module("unirna_tf")
        except ImportError as err:
            raise ImportError(
                "UniRNA adapter requires an installed " + chr(96) + "unirna_tf" + chr(96) + " package or an explicit "
                "encoder_factory=" + chr(96) + "<module>:<callable>" + chr(96) + " loader."
            ) from err

        if source is None:
            raise ValueError("UniRNA adapter requires " + chr(96) + "pretrained" + chr(96) + " pointing to a unirna_tf checkpoint directory.")
        if random_init:
            config = AutoConfig.from_pretrained(source, local_files_only=True)
            return AutoModel.from_config(config)
        return AutoModel.from_pretrained(source, **kwargs)
