# UniMol-Lite

UniMol-Lite is a simplified implementation of [Uni-Mol](https://github.com/dptech-corp/Uni-Mol).

## Installation

```shell
pip install ssh://git@github.com/macroversus/unimol_lite.git
# Download pre-trained models
wget http://ohgy1146835.bohrium.tech:50001/download/base_unimol -O base_unimol.zip
unzip base_unimol.zip
```

## Pre-trained models

The pre-trained model files are organized as follows:

```
base_unimol/
|-- config.json
`-- model.safetensors
```

## Usage examples

### 1. Prepare data

#### Option 1: use SMILES directly

```python
from unimol_lite.data import smiles2unimolitem

# convert a single SMILES
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item = smiles2unimolitem(smiles)
```

#### Option 2: use an LMDB dataset

1. Convert SMILES to LMDB:

```python
import pandas as pd

# read SMILES from a CSV file
df = pd.read_csv("examples/example_prepare_lmdb.csv")

# batch-convert with the script
python examples/prepare_lmdb.py -i input.csv -o output.lmdb -c SMILES_COLUMN
```

2. Load the data with LMDBDataset:

```python
from unimol_lite.data import LMDBDataset
from torch.utils.data import DataLoader

# create the dataset
dataset = LMDBDataset("output.lmdb")

# create the dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collater
)
```

#### Input data format
- SMILES: molecular strings in standard SMILES format
- CSV file: a CSV file containing a SMILES column; the column name can be set with `-c`

#### Intermediate data format
Each entry in the LMDB dataset contains:
```python
{
    "smiles": str,                # original SMILES
    "item": {
        "src_tokens": tensor,     # [L] token sequence of atom types
        "src_coord": tensor,      # [L,3] 3D atom coordinates
        "src_distance": tensor,   # [L,L] inter-atom distance matrix
        "src_edge_type": tensor,  # [L,L] inter-atom edge types
    }
}
```

#### Batched data format
A batch produced by the DataLoader contains:
```python
{
    "src_tokens": tensor,     # [B,L]
    "src_coord": tensor,      # [B,L,3]
    "src_distance": tensor,   # [B,L,L]
    "src_edge_type": tensor,  # [B,L,L]
}
```
where B is the batch size and L the sequence length (including padding).

### 2. Use the model

```python
from unimol_lite.model import UniMolModel, UniMolConfig

# initialize the configuration
config = UniMolConfig()

# load the pre-trained model
model = UniMolModel.from_pretrained("base_unimol")

# run inference
encoder_output, pair_output = model(**batch)
# encoder_output: [B,L,D] molecular representation
# pair_output: [B,L,L,D] atom-pair representation
# in general you will want encoder_rep[:, 0, :] as the whole-molecule embedding
```

### 3. Outputs

The model output has two parts:
1. encoder_output: per-molecule/per-atom representation, shape [batch_size, seq_len, hidden_dim]
2. pair_output: atom-pair representation, shape [batch_size, seq_len, seq_len, hidden_dim]

These representations can be used for:
- molecular property prediction
- molecular similarity computation
- molecular structure analysis
- and other downstream tasks

## Alternative download

If wget is unavailable, [click here](http://ohgy1146835.bohrium.tech:50001/base_unimol) for more download options.

## License

This project is released under the MIT license. See the [LICENSE](LICENSE) file.
