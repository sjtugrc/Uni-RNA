import numpy as np
import torch
from scipy.spatial.distance import squareform
from torch.nn.functional import pdist
from .smiles2coords import smiles2coords
from .utils import numpy_seed
from ..model.tokenizer import UniMolTokenizer

def generate_unimolitem(
    atoms: np.ndarray,
    coordinates: np.ndarray,
    tokenizer: UniMolTokenizer = UniMolTokenizer(),
    random_seed: int = 42,
    max_atoms: int = 256,
    normalize_coord: bool = True,
    *addl_seeds
) -> dict[str, torch.Tensor]:
    """Generate input dictionary for UniMol model.

    Args:
        atoms (np.ndarray): np.ndarray of atom symbols, shape (n_atoms,)
        coordinates (np.ndarray): np.ndarray of coordinates, shape (n_atoms, 3)
        tokenizer (UniMolTokenizer, optional): tokenizer object. Defaults to UniMolTokenizer().
        random_seed (int, optional): random seed effecting the random selection of atoms when atoms exceed max_atoms. Defaults to 42.
        max_atoms (int, optional): maximum number of atoms to consider. Defaults to 256.
        normalize_coord (bool, optional): whether to normalize the coordinates. Defaults to True.

    Returns:
        dict: dictionary containing the following keys:
            - src_tokens: tokenized atoms with cls and sep tokens
            - src_distance: distance matrix of atoms
            - src_coord: coordinates of atoms
            - src_edge_type: edge types of atoms
    """    
    assert len(atoms) == len(coordinates), "Atoms and coordinates shape mismatch"
    mask_hydrogen = atoms != "H"
    atoms = atoms[mask_hydrogen]
    coordinates = coordinates[mask_hydrogen]
    if max_atoms and len(atoms) > max_atoms:
        with numpy_seed(random_seed, *addl_seeds):
            index = np.random.choice(len(atoms), max_atoms, replace=False)
            atoms = np.array(atoms)[index]
            coordinates = coordinates[index]
    if normalize_coord:
        coordinates = coordinates - coordinates.mean(axis=0)
    src_tokens = torch.tensor(tokenizer.encode(atoms.tolist()), dtype=torch.long)
    src_tokens = torch.cat(
        [
            torch.tensor([tokenizer.cls_token_id]), 
            src_tokens,
            torch.tensor([tokenizer.sep_token_id])
        ], 
        dim=0
    )
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    coordinates = torch.cat(
        [
            torch.full_like(coordinates[0], 0.0).unsqueeze(0), 
            coordinates,
            torch.full_like(coordinates[0], 0.0).unsqueeze(0)
        ], 
        dim=0
    )
    num_types = len(tokenizer)
    node_input = src_tokens.clone()
    src_edges = node_input.view(-1, 1) * num_types + node_input.view(1, -1)
    distances = torch.from_numpy(squareform(pdist(coordinates.view(-1, 3))).astype(np.float32))
    return dict(
        src_tokens  = src_tokens,
        src_distance = distances,
        src_coord = coordinates,
        src_edge_type = src_edges,
    )

def smiles2unimolitem(
    smiles: str,
    tokenizer: UniMolTokenizer = UniMolTokenizer(),
    coord_pool_count: int = 1,
    random_seed: int = 42,
    max_atoms: int = 256,
    normalize_coord: bool = True,
    *addl_seeds
) -> dict[str, torch.Tensor]:
    """Generate input dictionary for UniMol model from smiles string.

    Args:
        smiles (str, optional): input smiles string.
        tokenizer (UniMolTokenizer, optional): tokenizer object. Defaults to UniMolTokenizer().
        coord_pool_count (int, optional): count of generated coordinates. Defaults to 1.
        random_seed (int, optional): random seed effecting coordinate selection in coord_pool, and random selection of atoms when atoms exceed max_atoms. Defaults to 42.
        max_atoms (int, optional): maximum number of atoms to consider. Defaults to 256.
        normalize_coord (bool, optional): whether to normalize the coordinates. Defaults to True.

    Returns:
        dict: dictionary containing the following keys:
            - src_tokens: tokenized atoms with cls and sep tokens
            - src_distance: distance matrix of atoms
            - src_coord: coordinates of atoms
            - src_edge_type: edge types of atoms
    """
    atoms, coordinates_list = smiles2coords(smiles, coord_pool_count)
    with numpy_seed(random_seed, *addl_seeds):
        sample_idx = np.random.randint(len(coordinates_list))
    coordinates = coordinates_list[sample_idx]
    return generate_unimolitem(
        atoms = atoms,
        coordinates = coordinates,
        tokenizer = tokenizer,
        random_seed = random_seed,
        max_atoms = max_atoms,
        normalize_coord = normalize_coord,
        *addl_seeds
    )