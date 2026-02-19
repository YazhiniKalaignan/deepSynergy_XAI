import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from chemopy import ChemoPy
import re
import sys

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

def generate_physico_df(smiles_file, labels_file):
    smiles_df = pd.read_csv(smiles_file, header=None, names=['Drug','SMILES'])
    labels_df = pd.read_csv(labels_file)
    labels_df.columns = labels_df.columns.str.strip()
    def preprocess(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.AddHs(mol) if mol else None
    smiles_df['Mol'] = smiles_df['SMILES'].apply(preprocess)
    cmp = ChemoPy(ignore_3D=True, include_fps=False)
    class DummyFile:
        def write(self, x): pass
        def flush(self): pass
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    drug_desc_map = {}
    for _, row in smiles_df.iterrows():
        mol = row['Mol']
        if mol is None:
            drug_desc_map[row['Drug']] = None
            continue
        desc_df = cmp.calculate([mol])
        drug_desc_map[row['Drug']] = desc_df.iloc[0].values
        if 'desc_names' not in locals():
            desc_names = list(desc_df.columns)
    sys.stdout = save_stdout
    combined_names = [f"A_{n}" for n in desc_names] + [f"B_{n}" for n in desc_names]
    labels_df['DrugA_desc'] = labels_df['drug_a_name'].map(drug_desc_map)
    labels_df['DrugB_desc'] = labels_df['drug_b_name'].map(drug_desc_map)
    labels_df['Combined_chem_desc'] = labels_df.apply(
        lambda row: np.concatenate([row['DrugA_desc'], row['DrugB_desc']])
        if row['DrugA_desc'] is not None and row['DrugB_desc'] is not None else None,
        axis=1
    )
    chem_desc_df = pd.DataFrame(labels_df['Combined_chem_desc'].tolist(), columns=combined_names)
    return chem_desc_df

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

def process_cellline_matrix(matrix_path, annot_path, target_cells):
    def normalize(x):
        if pd.isna(x): return ""
        x = str(x).strip().upper()
        return re.sub(r'[^A-Z0-9]', '', x)  
    expr = pd.read_csv(matrix_path, index_col=0, engine="python")
    annot = pd.read_csv(annot_path)
    expr.columns = [normalize(c) for c in expr.columns]
    target_norm = [normalize(x) for x in target_cells]
    mapping = {}
    for col in expr.columns:
        for tnorm, toriginal in zip(target_norm, target_cells):
            if tnorm in col:
                mapping[col] = toriginal
    expr_39 = expr[list(mapping.keys())].copy()
    expr_39.columns = [mapping[c] for c in expr_39.columns]
    expr_39 = expr_39.groupby(expr_39.columns, axis=1).mean()
    return expr_39