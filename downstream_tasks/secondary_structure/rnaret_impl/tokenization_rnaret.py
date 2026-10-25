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


"""k-mer tokenizer for RNAret (mirrors upstream ``seq_tokenizer`` in ssp.py).

Vocabulary
----------
Upstream uses a hand-rolled vocabulary:

    id 0: PAD
    id 1: flank / "BOS-like" (padded on both sides for k>1 to align k-mer
          window with nucleotide positions; k=1 uses no flanks)
    id 2: UNK (any kmer containing a base not in {A, T, C, G})
    ids 6..(4^k + 5): k-mers in lexicographic order over ``ATCG``
                      (``itertools.product('ATCG', repeat=k)``).

Sequences are upper-cased and ``U`` is replaced with ``T`` before
tokenization (upstream does this unconditionally — not opt-in).

For k=1 the window has length 1 so no flanking tokens are emitted: the
tokenized sequence has length ``len(seq)`` with no CLS/EOS.

This class exposes the small ``__call__`` / ``encode`` surface MM dataset
tokenization needs from an ``AutoTokenizer``.
"""

from __future__ import annotations

import itertools
from typing import Sequence


class RNAretKmerTokenizer:
    """HuggingFace-compatible k-mer tokenizer for RNAret.

    Only the small slice of the ``PreTrainedTokenizer`` surface that
    ``RNADataset`` / ``RNACollator`` actually uses is implemented:

        - ``tokenizer(text, max_length=..., truncation=False,
          return_attention_mask=True, return_tensors=None)`` →
          ``{"input_ids": list[int], "attention_mask": list[int]}``
        - ``tokenizer.encode(text, add_special_tokens=False)`` →
          ``list[int]``
        - ``tokenizer.pad_token_id`` → 0

    The ``add_special_tokens`` flag is accepted for interface parity but
    is a no-op at k=1 (no CLS/EOS in upstream); for k>1 the flanking "1"
    tokens added by the upstream k-mer window act as CLS/EOS analogues
    and are always emitted.
    """

    pad_token_id: int = 0
    unk_token_id: int = 2
    flank_token_id: int = 1

    # HF tokenizer interface parity: RNAret has no CLS/SEP/EOS, but MM's `Dataset`
    # reads these attributes directly (assuming a standard HF tokenizer), so expose
    # them as None rather than relying on the caller to guard with `hasattr`.
    cls_token = None
    bos_token = None
    sep_token = None
    eos_token = None
    model_max_length: int = int(1e30)

    def __init__(self, k: int = 1):
        if k < 1:
            raise ValueError(f"k-mer size must be >= 1, got {k}")
        self.k = k
        kmer_list = ["".join(p) for p in itertools.product("ATCG", repeat=k)]
        # ids 0..5 reserved (PAD, flank, UNK, + 3 others held by upstream
        # for pretrain masking tokens); k-mers start at id 6.
        self.kmer_to_index: dict[str, int] = {kmer: idx + 6 for idx, kmer in enumerate(kmer_list)}
        self.vocab_size: int = 4**k + 6

    # ----------------------------------------------------------------- core
    def _tokenize(self, seq: str) -> list[int]:
        seq = seq.upper().replace("U", "T")
        seq_len = len(seq)
        k = self.k
        tokens: list[int] = [0] * seq_len
        kmers = [seq[i : i + k] for i in range(seq_len - k + 1)]
        indices = [self.kmer_to_index.get(kmer, self.unk_token_id) for kmer in kmers]
        # Place k-mer ids in the centre of the token strip; pad both
        # sides with the flank token so every nucleotide maps to one
        # output position.  At k=1 both padding regions are empty.
        left_pad = k // 2
        right_pad = (k - 1) // 2
        for i, idx in enumerate(indices):
            tokens[left_pad + i] = idx
        for i in range(left_pad):
            tokens[i] = self.flank_token_id
        for i in range(right_pad):
            tokens[left_pad + len(indices) + i] = self.flank_token_id
        return tokens

    # ---------------------------------------------------------------- HF api
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # The k-mer window already adds flanks for k>1; there are no
        # separate CLS/SEP tokens to prepend/append, so this flag is a
        # no-op and exists only for AutoTokenizer interface parity.
        del add_special_tokens
        return self._tokenize(text)

    def __call__(
        self,
        text: str | Sequence[str],
        max_length: int | None = None,
        truncation: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> dict:
        # Accept and ignore extra HuggingFace tokenizer kwargs (e.g. `padding`) that callers such
        # as infer.py pass for interface parity.
        del add_special_tokens, kwargs  # see encode()
        # MM's `Dataset.collate` tokenizes a whole column at once, so `text` may be a single
        # string (DeepRNASS RNADataset path) or a list of strings (MM batched path). Handle both.
        is_batched = not isinstance(text, str)
        sequences = list(text) if is_batched else [text]
        input_ids: list[list[int]] = []
        for sequence in sequences:
            ids = self._tokenize(sequence)
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]
            input_ids.append(ids)
        result: dict = {"input_ids": input_ids if is_batched else input_ids[0]}
        if return_attention_mask:
            masks = [[1] * len(ids) for ids in input_ids]
            result["attention_mask"] = masks if is_batched else masks[0]
        if return_tensors is not None:
            import torch

            if is_batched:
                result = {k: [torch.tensor(v, dtype=torch.long) for v in vals] for k, vals in result.items()}
            else:
                result = {k: torch.tensor([v], dtype=torch.long) for k, v in result.items()}
        return result
