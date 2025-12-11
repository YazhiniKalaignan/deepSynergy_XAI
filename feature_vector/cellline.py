import pandas as pd
import re
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
