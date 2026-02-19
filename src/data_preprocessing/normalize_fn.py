import os
import pickle
import gzip
import numpy as np
import pandas as pd
def load(norm='tanh_norm', test_fold=0, val_fold=1):
    def normalize(X,means1=None,std1=None,means2=None,std2=None,feat_filt=None,norm='tanh_norm',eps=1e-8):
        if std1 is None:
            std1_raw = np.nanstd(X, axis=0)
            feat_filt = std1_raw > eps
            X = X[:, feat_filt]
            std1 = std1_raw[feat_filt]
            means1 = np.nanmean(X, axis=0)
        else:
            X = X[:, feat_filt]  
        X = np.ascontiguousarray(X)
        X = (X - means1) / (std1 + eps)
        X[~np.isfinite(X)] = 0.0 
        if norm == 'norm':
            return X, means1, std1, feat_filt
        X = np.tanh(X)
        X[~np.isfinite(X)] = 0.0
        if norm == 'tanh':
            return X, means1, std1, feat_filt
        if means2 is None:
            means2 = np.nanmean(X, axis=0)
        if std2 is None:
            std2 = np.nanstd(X, axis=0)
        X = (X - means2) / (std2 + eps)
        X[~np.isfinite(X)] = 0.0
        return X, means1, std1, means2, std2, feat_filt
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_folder = os.path.join(BASE_DIR, "datasets")
    X_file = os.path.join(data_folder, "Feature_vectors_dataset.p.gz")
    labels_file = os.path.join(data_folder, "labels.csv")
    with gzip.open(X_file, 'rb') as f:
        X = pickle.load(f)
    labels = pd.read_csv(labels_file, index_col=0)
    idx_tr = np.where((labels['fold'] != test_fold) & (labels['fold'] != val_fold))[0]
    idx_val = np.where(labels['fold'] == val_fold)[0]
    idx_train = np.where(labels['fold'] != test_fold)[0]
    idx_test = np.where(labels['fold'] == test_fold)[0]

    X_tr = X.iloc[idx_tr]
    X_val = X.iloc[idx_val]
    X_train = X.iloc[idx_train]
    X_test = X.iloc[idx_test]

    y_tr = labels.iloc[idx_tr]['synergy'].values
    y_val = labels.iloc[idx_val]['synergy'].values
    y_train = labels.iloc[idx_train]['synergy'].values
    y_test = labels.iloc[idx_test]['synergy'].values

    X_tr_np = X_tr.select_dtypes(include=[np.number]).to_numpy(np.float32)
    X_val_np = X_val.select_dtypes(include=[np.number]).to_numpy(np.float32)
    X_train_np = X_train.select_dtypes(include=[np.number]).to_numpy(np.float32)
    X_test_np = X_test.select_dtypes(include=[np.number]).to_numpy(np.float32)

    y_tr_np = y_tr.astype(np.float32).reshape(-1, 1)
    y_val_np = y_val.astype(np.float32).reshape(-1, 1)
    y_train_np = y_train.astype(np.float32).reshape(-1, 1)
    y_test_np = y_test.astype(np.float32).reshape(-1, 1)

    if norm == "tanh_norm":
        X_tr_np, mean1, std1, mean2, std2, feat_filt = normalize(X_tr_np, norm=norm)
        X_val_np, _, _, _, _, _ = normalize(
            X_val_np, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm
        )
    else:
        X_tr_np, mean1, std1, feat_filt = normalize(X_tr_np, norm=norm)
        X_val_np, _, _, _ = normalize(
            X_val_np, mean1, std1, feat_filt=feat_filt, norm=norm)
    return (X_tr_np,X_val_np,X_train_np,X_test_np,y_tr_np,y_val_np,y_train_np,y_test_np)



