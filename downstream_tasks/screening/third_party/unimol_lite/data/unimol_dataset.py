from torch.utils.data import Dataset
import lmdb
from pathlib import Path
import pickle
from functools import lru_cache
from . import unimol_collate_fn, smiles2unimolitem
from ..model import UniMolTokenizer


class BaseDataset(Dataset):
    def __init__(self, smiles_lists: list, tokenizer: UniMolTokenizer = UniMolTokenizer()):
        self.smiles_lists = smiles_lists
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles_lists)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return smiles2unimolitem(self.smiles_lists[idx])

    def collater(self, batch):
        return unimol_collate_fn(batch, self.tokenizer)


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path: str, included_keys: list | None = None, tokenizer: UniMolTokenizer = UniMolTokenizer()):
        assert Path(lmdb_path).exists(), f"{lmdb_path} does not exist"
        self.env = lmdb.open(
            lmdb_path, 
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if included_keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        else:
            assert all(isinstance(key, bytes) for key in included_keys), "Keys must be bytes"
            self._keys = included_keys
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)["item"]
        return data

    def collater(self, batch):
        return unimol_collate_fn(batch, self.tokenizer)
    