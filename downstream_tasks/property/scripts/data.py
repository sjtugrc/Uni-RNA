import pandas as pd
from typing import Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch

def read_data(path: str, seq_col: str = 'sequence', label_cols: List[str] = None) -> Dict[str, list]:
    """
    Unified csv/pkl reader with customizable sequence and label column names.
    """
    if not label_cols:
        label_cols = ['label']

    # dispatch by file suffix (pkl branch)
    if path.endswith('.pkl'):
        # numpy1 compatibility for pickles created under numpy2 (numpy._core -> numpy.core)
        try:
            import numpy._core  # noqa: F401
        except ImportError:
            import numpy.core, numpy.core.multiarray, sys
            sys.modules.setdefault('numpy._core', numpy.core)
            sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    seqs = df[seq_col].tolist()
    
    if len(label_cols) == 1:
        labels = df[label_cols[0]].tolist()
    else:
        labels = df[label_cols].values.tolist()
    
    return {
        'sequence': seqs,
        'label': labels
    }

def scale_data(data: Dict[str, list], scaler: Optional[StandardScaler] = None):
    """
    Scale list-shaped labels with a StandardScaler.
    Pass scaler=None for the training set (fit a new scaler); pass a fitted
    scaler for val/test sets.
    """
    import numpy as np
    labels_np = np.array(data['label'])
    is_1d = labels_np.ndim == 1
    if is_1d:
        labels_np = labels_np.reshape(-1, 1)
    
    if scaler is None:
        scaler = StandardScaler()
        labels_scaled = scaler.fit_transform(labels_np)
    else:
        labels_scaled = scaler.transform(labels_np)
        
    if is_1d:
        labels_scaled = labels_scaled.flatten()
        
    return labels_scaled.tolist(), scaler

class RNADataset(Dataset):
    """
    Dataset holding raw text only; tokenization is deferred to the model
    forward pass (late tokenization).
    """
    def __init__(self, data_dict: Dict[str, list]):
        self.sequences = data_dict['sequence']
        self.labels = data_dict['label']
        assert len(self.sequences) == len(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate keeps raw sequence strings for flexible per-model tokenization;
    labels are stacked into a tensor.
    """
    sequences = [item['sequence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    return {
        'sequence': sequences,
        'label': labels
    }
