import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
def generate_ecfp6_df(smiles_file, radius=6, nBits=1024):
    smiles_df = pd.read_csv(smiles_file, header=None, names=["Drug", "SMILES"]) 
    def generate_ecfp6(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    fingerprints = []
    for _, row in smiles_df.iterrows():
        fp = generate_ecfp6(row["SMILES"])
        if fp is not None:
            fingerprints.append([row["Drug"]] + fp.tolist())
    columns = ["Drug"] + [f"ECFP6_{i}" for i in range(nBits)]
    return pd.DataFrame(fingerprints, columns=columns)


