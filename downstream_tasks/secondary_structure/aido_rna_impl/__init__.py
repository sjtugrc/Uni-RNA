"""Vendored genbio AIDO.RNA (RNABert) architecture for the MM-native runner.

genbio publishes AIDO.RNA checkpoints as *weights only* (``model_type: "rnabert"``,
no remote code), and that ``model_type`` collides with multimolecule's own RnaBert —
so ``AutoModel.from_pretrained("genbio-ai/AIDO.RNA-1.6B")`` silently builds the wrong
architecture (multimolecule RnaBert: GELU + learned positions) instead of genbio's
SwiGLU + RoPE RNABert. To run AIDO.RNA faithfully we reuse genbio's own modeling code,
copied verbatim from ``genbio-ai/ModelGenerator`` (``huggingface/aido.rna/aido_rna/models``),
and load the vendored classes directly to bypass the registry collision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .configuration_rnabert import RNABertConfig
from .modeling_rnabert import RNABertForMaskedLM, RNABertModel
from .tokenization_rnabert import RNABertTokenizer

# genbio's HF repos ship only config + weights (no ``vocab.txt``); the real vocabulary
# lives in their package, so we vendor it next to the modeling code.
VOCAB_FILE = str(Path(__file__).resolve().parent / "vocab.txt")


def load_aido_rna_model(source: str, **kwargs: Any) -> RNABertModel:
    """Load genbio's RNABert (AIDO.RNA) directly, bypassing the ``rnabert`` collision."""
    return RNABertModel.from_pretrained(source, **kwargs)


def load_aido_rna_tokenizer(vocab_file: str | None = None, **kwargs: Any) -> RNABertTokenizer:
    return RNABertTokenizer(vocab_file=vocab_file or VOCAB_FILE, **kwargs)


__all__ = [
    "RNABertConfig",
    "RNABertModel",
    "RNABertForMaskedLM",
    "RNABertTokenizer",
    "VOCAB_FILE",
    "load_aido_rna_model",
    "load_aido_rna_tokenizer",
]
