import pandas as pd

# from collections import Counter
# import os
import scanpy as sc
from sklearn.model_selection import train_test_split
from anndata._core.anndata import AnnData
from typing import Dict, Tuple

# import numpy as np


def split_train_test(
    adata: AnnData,
    out_dir: str,
    min_cells: int=20,
    min_samples: int=3,
    test_pt_size: float=0.2,
    random_state: int=3467,
    verbose: int=0,
) -> Tuple[Dict[str, AnnData], Dict[str, AnnData]]:
    # split dataset by cell type-----------------------------
    # for each cell type,
    ## filter patients with # of cells < 20
    ## if # of patients in each group >= 3:
    ## Dowsample and upsample(SMOTE) # of cells in each patient to Median value.

    adata_train_dict = {}
    adata_test_dict = {}
    adata_train_meta = pd.DataFrame(columns=adata.obs.columns)
    adata_test_meta = pd.DataFrame(columns=adata.obs.columns)

    celltypes_all = adata.obs["ct"].astype("category").cat.categories.tolist()
    celltypes_exclude = []

    for ct in celltypes_all:
        if verbose > 0:
            print("Start working on", ct)

        adata_ct = adata[adata.obs["ct"] == ct,]
        # Number of cells in each patient
        n_cell_pt = adata_ct.obs.groupby(
            ["y", "patient_id"], observed=True, as_index=False
        ).size()
        # Remove paients with cells less than min_cells
        pt_keep = n_cell_pt.patient_id[n_cell_pt["size"] >= min_cells].tolist()

        # Cell types with 0 patient has cells >= min_cells
        if len(pt_keep) == 0:
            # print(ct, f': all patients has < {min_cells} cells')
            celltypes_exclude.append(ct)
            continue

        adata_ct = adata_ct[adata_ct.obs["patient_id"].isin(pt_keep),]

        n_cell_pt = adata_ct.obs.groupby(
            ["y", "patient_id"], observed=True, as_index=False
        ).size()

        # Skip cell types with less than 3 patients in at least one condition
        if (n_cell_pt.y.nunique() < 2) | (
            (n_cell_pt.y.value_counts() < min_samples).any()
        ):
            # print(ct, 'has less than 3 patients in condition')
            celltypes_exclude.append(ct)
            continue

        # Remove lowly-expressed genes
        sc.pp.filter_genes(adata_ct, min_cells=(adata_ct.n_obs) * 0.1)
        sc.pp.highly_variable_genes(adata_ct, n_top_genes=2000)
        adata_ct = adata_ct[:, adata_ct.var.highly_variable]

        # Split data into train and test set--------------------------
        pat_meta_temp = n_cell_pt
        cell_meta_temp = adata_ct.obs

        # split data at patient-level
        rest_, patient_test_id_list = train_test_split(
            pat_meta_temp.patient_id,
            test_size=test_pt_size,
            stratify=pat_meta_temp.y,
            random_state=random_state,
        )

        # print(len(patient_test_id_list),'patients in test set')
        # print(len(rest_), 'patients remaining...')

        # retrieve cell-level index for train and test set
        test_idx = cell_meta_temp[
            cell_meta_temp["patient_id"].isin(patient_test_id_list)
        ].index.tolist()
        train_idx = cell_meta_temp[
            cell_meta_temp["patient_id"].isin(rest_)
        ].index.tolist()

        # get adata object for train and test
        train_adata = adata_ct[train_idx, :]
        test_adata = adata_ct[test_idx, :]

        # Scale training data
        sc.pp.scale(train_adata, max_value=10)
        scaler_stat = train_adata.var[["mean", "std"]]
        train_adata.var = scaler_stat

        adata_train_dict[ct] = train_adata
        adata_test_dict[ct] = test_adata

        # Summarize metadata
        adata_train_meta = pd.concat([adata_train_meta, train_adata.obs])
        adata_test_meta = pd.concat([adata_test_meta, test_adata.obs])

        adata_train_meta.to_csv(f"{out_dir}/Meta_all_train.csv", index=True)
        adata_test_meta.to_csv(f"{out_dir}/Meta_all_test.csv", index=True)

    print("Following cell types are excluded since no enough cells/patients:")
    print(*celltypes_exclude, sep=", ")

    return adata_train_dict, adata_test_dict
