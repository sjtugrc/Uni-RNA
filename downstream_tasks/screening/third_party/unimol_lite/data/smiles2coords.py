import numpy as np
from typing import Tuple, List
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def mol2coords_2d(mol) -> np.ndarray:
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {Chem.MolToSmiles(mol)}"
    return coordinates


def mol2coords_3d(mol, count) -> List[np.ndarray]:
    coordinate_list = []
    for seed in range(count):
        try:
            res = AllChem.EmbedMolecule(
                mol, randomSeed=seed
            )  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                AllChem.MMFFOptimizeMolecule(mol)  # some conformer can not use MMFF optimize
                coordinates = mol.GetConformer().GetPositions()
            elif res == -1:
                mol_tmp = deepcopy(mol)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                AllChem.MMFFOptimizeMolecule(mol_tmp)  # some conformer can not use MMFF optimize
                coordinates = mol_tmp.GetConformer().GetPositions()
            else:
                ...
        except:
            # print("Failed to generate 3D, replace with 2D")
            coordinates = mol2coords_2d(mol)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {Chem.MolToSmiles(mol)}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def smiles2coords(smiles: str, count: int = 10) -> Tuple[np.ndarray, list]:
    """_summary_

    Args:
        smiles (str): smiles string
        count (int, optional): conformer num,all==11, 10 * 3d + 1 * 2d. Defaults to 10.

    Returns:
        Tuple[np.ndarray, list]: atoms, coordinates list
    """    
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = AllChem.AddHs(mol)
        if len(mol.GetAtoms()) > 400:
            coordinate_list = [mol2coords_2d(mol)] * (count + 1)
            # print("atom num >400,use 2D coords", smi)
        else:
            coordinate_list = mol2coords_3d(mol, count)
            # add 2d conf
            coordinate_list.append(mol2coords_2d(mol).astype(np.float32))
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
        return np.array(atoms), coordinate_list
    except:
        return [None, None]
