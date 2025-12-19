import pandas as pd
import numpy as np
from rdkit import Chem
def generate_toxico_df(smiles_file, alerts_file):
    smiles_df = pd.read_csv(smiles_file, header=None, names=["Drug","SMILES"])
    smiles_df["Mol"] = smiles_df["SMILES"].apply(lambda x: Chem.MolFromSmiles(x))
    alerts_df = pd.read_csv(alerts_file)
    alerts_df = alerts_df[["description","smarts"]].rename(columns={"description":"Name","smarts":"SMARTS"}) 
    def safe_smarts(smarts):
        try:
            return Chem.MolFromSmarts(smarts)
        except:
            return None
    alerts_df["Mol"] = alerts_df["SMARTS"].apply(safe_smarts)
    alerts_df = alerts_df[alerts_df["Mol"].notnull()].reset_index(drop=True)
    feature_matrix = np.zeros((len(smiles_df), len(alerts_df)), dtype=int)
    for i, drug_mol in enumerate(smiles_df["Mol"]):
        for j, alert_mol in enumerate(alerts_df["Mol"]):
            if drug_mol.HasSubstructMatch(alert_mol):
                feature_matrix[i,j] = 1
    tox_df = pd.DataFrame(feature_matrix, columns=alerts_df["Name"])
    tox_df["Drug"] = smiles_df["Drug"]
    return tox_df



