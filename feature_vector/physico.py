import pandas as pd
import numpy as np
from rdkit import Chem
from chemopy import ChemoPy
import sys
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
    chem_desc_df['synergy'] = labels_df['synergy']
    return chem_desc_df
