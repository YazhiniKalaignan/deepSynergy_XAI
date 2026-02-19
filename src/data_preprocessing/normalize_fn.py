import os
import pickle
import gzip
import numpy as np
import pandas as pd
def load(norm='tanh_norm', test_fold=0, val_fold=1):
    def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm', eps=1e-8):
        if std1 is None:
            std1_raw  = np.nanstd(X, axis=0)
            feat_filt = std1_raw > eps
            X         = X[:, feat_filt]
            std1      = std1_raw[feat_filt]
            means1    = np.nanmean(X, axis=0)
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
    BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_folder   = os.path.join(BASE_DIR, "datasets")
    features_file = os.path.join(data_folder, "Feature_vectors_dataset.p.gz")
    labels_file   = os.path.join(data_folder, "labels.csv")
    with gzip.open(features_file, 'rb') as f:
        features_df = pickle.load(f)

    labels_df = pd.read_csv(labels_file, index_col=0)

    #Dropped synergy score and other non feature columns
    non_feature_cols = [c for c in ['Unnamed: 0', 'synergy', 'fold','drug_a_name', 'drug_b_name', 'cell_line']
                       if c in features_df.columns]
    features_df = features_df.drop(columns=non_feature_cols)

    train_idx = np.where((labels_df['fold'] != test_fold)&(labels_df['fold'] != val_fold))[0]
    val_idx = np.where(labels_df['fold'] == val_fold)[0]
    all_train_idx = np.where(labels_df['fold'] != test_fold)[0]
    test_idx = np.where(labels_df['fold'] == test_fold)[0]

    train_features = features_df.iloc[train_idx].to_numpy(np.float32)
    val_features = features_df.iloc[val_idx].to_numpy(np.float32)
    all_train_features = features_df.iloc[all_train_idx].to_numpy(np.float32)
    test_features = features_df.iloc[test_idx].to_numpy(np.float32)

    train_targets = labels_df.iloc[train_idx]['synergy'].values.astype(np.float32).reshape(-1, 1)
    val_targets = labels_df.iloc[val_idx]['synergy'].values.astype(np.float32).reshape(-1, 1)
    all_train_targets = labels_df.iloc[all_train_idx]['synergy'].values.astype(np.float32).reshape(-1, 1)
    test_targets = labels_df.iloc[test_idx]['synergy'].values.astype(np.float32).reshape(-1, 1)

    if norm == "tanh_norm":
        train_features,means1, std1, means2, std2, feat_filt = normalize(train_features, norm=norm)
        val_features,_, _, _, _, _ = normalize(val_features,means1, std1, means2, std2, feat_filt=feat_filt, norm=norm)
        all_train_features,_, _, _, _, _ = normalize(all_train_features, means1, std1, means2, std2, feat_filt=feat_filt, norm=norm)
        test_features,_, _, _, _, _ = normalize(test_features, means1, std1, means2, std2, feat_filt=feat_filt, norm=norm)
    else:
        train_features, means1, std1, feat_filt = normalize(train_features, norm=norm)
        val_features, _, _, _ = normalize(val_features, means1, std1, feat_filt=feat_filt, norm=norm)
        all_train_features, _, _, _ = normalize(all_train_features, means1, std1, feat_filt=feat_filt, norm=norm)
        test_features, _, _, _ = normalize(test_features, means1, std1, feat_filt=feat_filt, norm=norm)

    return (train_features, val_features, all_train_features, test_features, train_targets,  val_targets,  all_train_targets,  test_targets)