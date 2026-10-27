import os
from typing import List, Optional
from transformers.tokenization_utils import PreTrainedTokenizer

VOCAB_LIST = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[UNK]",
    "C",
    "N",
    "O",
    "S",
    "H",
    "Cl",
    "F",
    "Br",
    "I",
    "Si",
    "P",
    "B",
    "Na",
    "K",
    "Al",
    "Ca",
    "Sn",
    "As",
    "Hg",
    "Fe",
    "Zn",
    "Cr",
    "Se",
    "Gd",
    "Au",
    "Li",
    "[MASK]",
]


class UniMolTokenizer(PreTrainedTokenizer):
    """
    Constructs an UniMol tokenizer.
    """

    def __init__(
        self,
        unk_token="[UNK]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        sep_token="[SEP]",
        **kwargs,
    ):
        self.all_tokens = VOCAB_LIST
        self._id_to_token = dict(enumerate(self.all_tokens))
        self._token_to_id = {tok: ind for ind, tok in enumerate(self.all_tokens)}

        super().__init__(unk_token=unk_token, cls_token=cls_token, pad_token=pad_token, sep_token=sep_token, **kwargs)

        # TODO, all the tokens are added? But they are also part of the vocab... bit strange.
        # none of them are special, but they all need special splitting.

        self.unique_no_split_tokens = self.all_tokens
        self._update_trie(self.unique_no_split_tokens)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def _tokenize(self, text, **kwargs):
        return text.split()

    def get_vocab(self):
        base_vocab = self._token_to_id.copy()
        base_vocab.update(self.added_tokens_encoder)
        return base_vocab

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id.get(self.unk_token))

    def id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, self.unk_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return token_ids_0  # Multiple inputs always have an EOS token

    def get_special_tokens_mask(
        self,
        token_ids_0: List,
        token_ids_1: Optional[List] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return [1 if token in self.all_special_ids else 0 for token in token_ids_0]
        mask = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1) + [1]
        return mask

    def save_vocabulary(self, save_directory, filename_prefix):
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(self.all_tokens))
        return (vocab_file,)

    @property
    def vocab_size(self) -> int:
        return len(self.all_tokens)
