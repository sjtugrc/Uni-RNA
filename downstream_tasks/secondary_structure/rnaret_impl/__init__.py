"""Optional RNAret implementation used by the MM sequence adapter.

The heavy RetNet modules depend on optional ``torchscale`` / ``fairscale``.
Import concrete submodules directly so tokenizer-only use does not eagerly
require those packages.
"""

__all__ = ["RNARET_SSP_HPARAMS", "RNAretBackbone", "RNAretKmerTokenizer", "load_rnaret_model"]
