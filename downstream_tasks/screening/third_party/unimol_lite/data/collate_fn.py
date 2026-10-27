from typing import List, Dict
from torch import Tensor
from ..model import UniMolTokenizer

def collate_tokens(
    values: List[Tensor],
    pad_idx: int,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
) -> Tensor:
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src: Tensor, dst: Tensor):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

def collate_tokens_coords(
    values: List[Tensor],
    pad_idx: int | float,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
) -> Tensor:
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src: Tensor, dst: Tensor):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :, :] if left_pad else res[i][: len(v), :])
    return res

def collate_cross_2d(
    values: List[Tensor],
    pad_idx: int | float,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
) -> Tensor:
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src: Tensor, dst: Tensor):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(
            v,
            res[i][size_h - v.size(0) :, size_w - v.size(1) :]
            if left_pad
            else res[i][: v.size(0), : v.size(1)],
        )
    return res

def unimol_collate_fn(batch: List[Dict[str, any]], tokenizer: UniMolTokenizer = UniMolTokenizer()) -> Dict[str, Tensor]:
    """_summary_

    Args:
        batch (List[Dict[str, any]]): list of dict(s) containing the following keys: src_tokens, src_coord, src_distance, src_edge_type
            - src_tokens: tokenized atoms with cls and sep tokens
            - src_distance: distance matrix of atoms
            - src_coord: coordinates of atoms
            - src_edge_type: edge types of atoms
        tokenizer (UniMolTokenizer, optional): tokenizer for tokenizing atoms. Defaults to UniMolTokenizer().

    Returns:
        Dict[str, Tensor]: _description_
    """    
    new_batch = {}
    new_batch["src_tokens"] = collate_tokens(
        [x["src_tokens"] for x in batch], 
        pad_idx=tokenizer.pad_token_id, 
        left_pad=False, 
        pad_to_multiple=8
    )
    new_batch["src_coord"] = collate_tokens_coords(
        [x["src_coord"] for x in batch], 
        pad_idx=0, 
        left_pad=False, 
        pad_to_multiple=8
    )
    new_batch["src_distance"] = collate_cross_2d(
        [x["src_distance"] for x in batch], 
        pad_idx=0, 
        left_pad=False, 
        pad_to_multiple=8
    )
    new_batch["src_edge_type"] = collate_cross_2d(
        [x["src_edge_type"] for x in batch], 
        pad_idx=0, 
        left_pad=False, 
        pad_to_multiple=8
    )
    return new_batch