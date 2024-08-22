# import anndata
import itertools
import os
import pickle
import time

import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# from sklearn.preprocessing import LabelEncoder
from sklearn import svm

# import scanpy as sc
# import numpy as np
# import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .SVMRFECV import RFE, RFECV
from .utils_func import *
from anndata._core.anndata import AnnData
from matplotlib.axes._axes import Axes
from scpanel.SVMRFECV import RFECV
from typing import List, Optional, Tuple


def split_n_folds(adata_train: AnnData, nfold: int, out_dir: Optional[str]=None, random_state: int=2349) -> Tuple[List[List[int]], List[List[int]], List[List[float]]]:
    ## add: exclude patients without selected cell type
    n_cell_pat = adata_train.obs.groupby(["patient_id"])["ct"].count()
    exclude_pat = adata_train.obs["patient_id"].isin(n_cell_pat[n_cell_pat == 0].index)
    adata_train = adata_train[~exclude_pat]

    if sum(exclude_pat) > 0:
        print(
            n_cell_pat[n_cell_pat == 0].index.tolist(),
            "get excluded since no selected cell type appears",
        )

    ## split patients
    pat_meta_temp = adata_train.obs[["y", "patient_id"]].drop_duplicates().reset_index()
    cell_meta_temp = adata_train.obs.reset_index()

    patient_class = pat_meta_temp["y"].to_numpy()
    patient = pat_meta_temp["patient_id"].to_numpy()

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_state)
    # sss = StratifiedShuffleSplit(n_splits=nfold, test_size = test_size)

    patient_train_id_list = []
    patient_val_id_list = []

    train_patient_list = []
    val_patient_list = []

    train_index_list = []
    val_index_list = []

    weight_list = []

    for train_index, val_index in skf.split(patient, patient_class):

        train_patient_list.append(train_index)
        val_patient_list.append(val_index)

        patient_train_id = patient[train_index]
        patient_val_id = patient[val_index]

        patient_train_id_list.append(patient_train_id)
        patient_val_id_list.append(patient_val_id)

        cell_meta_fold_train = cell_meta_temp[
            cell_meta_temp["patient_id"].isin(patient_train_id)
        ]
        cell_meta_fold_test = cell_meta_temp[
            cell_meta_temp["patient_id"].isin(patient_val_id)
        ]

        # compute weight for each cell in each fold's training set
        w_fold_train = compute_cell_weight(cell_meta_fold_train)
        weight_list.append(w_fold_train.tolist())

        # get positional index for train and test set in each fold
        cell_train_id = cell_meta_fold_train.index.tolist()
        cell_val_id = cell_meta_fold_test.index.tolist()

        # cell_train_id.sort()
        # cell_val_id.sort()

        if cell_train_id not in train_index_list:
            train_index_list.append(cell_train_id)

        if cell_val_id not in val_index_list:
            val_index_list.append(cell_val_id)

    ## check if weights (np.Series) have the same order as train_index_list
    # np.array_equiv([idx for fold in train_index_list for idx in fold],
    #                w_fold_train.index.values)

    if out_dir is not None:
        # Output
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ## Data and index
        X_train, y_train = get_X_y_from_ann(adata_train)
        with open(os.path.join(out_dir, "Data_X_y.pkl"), "wb") as f:
            d = {"features": X_train, "labels": y_train}
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            del d

        with open(
            os.path.join(out_dir, "Data_" + str(nfold) + "fold_index.pkl"), "wb"
        ) as f:
            d = {
                "train": train_index_list,
                "val": val_index_list,
                "sample_weight": w_fold_train,
            }
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            del d

        ## nfold splitting information
        # get n_cells, patient_id and class prop for each set
        all_list = train_index_list + val_index_list
        n_cells = [len(sublist) for sublist in all_list]

        patient_ids = patient_train_id_list + patient_val_id_list

        class_prop = [
            np.unique(y_train[index], return_counts=True)[1] for index in all_list
        ]

        patient_prop_train = [
            pat_meta_temp.loc[fold].y.value_counts().tolist()
            for fold in train_patient_list
        ]
        patient_prop_val = [
            pat_meta_temp.loc[fold].y.value_counts().tolist()
            for fold in val_patient_list
        ]
        patient_prop = patient_prop_train + patient_prop_val

        nfold_info = pd.DataFrame(
            {
                "n_cells": n_cells,
                "patient_ids": patient_ids,
                "class_prop": class_prop,
                "pt_prop": patient_prop,
            }
        )

        train_col = ["train_f" + str(i) for i in range(1, nfold + 1)]
        val_col = ["val_f" + str(i) for i in range(1, nfold + 1)]
        nfold_info.index = train_col + val_col
        nfold_info.to_csv(f"{out_dir}/split_nfold_info.csv")

    return train_index_list, val_index_list, weight_list


def gene_score(
    adata_train: AnnData,
    train_index_list: List[List[int]],
    val_index_list: List[List[int]],
    sample_weight_list: List[List[float]],
    out_dir: str,
    ncpus: int,
    step: float=0.03,
    metric: str="average_precision",
    verbose: bool=False,
) -> Tuple[AnnData, RFECV]:

    # metric: https://scikit-learn.org/stable/modules/model_evaluation.html

    X, y = get_X_y_from_ann(adata_train)

    # Fill NaN in numpy
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    # model------------
    # model = svm.SVC(kernel="linear", class_weight = 'balanced', verbose=verbose, random_state = 123)
    model = svm.SVC(kernel="linear", verbose=verbose, random_state=123)

    rfecv = RFECV(
        estimator=model, step=step, scoring=metric, cv=10, n_jobs=ncpus, verbose=0
    )
    # X = StandardScaler().fit_transform(X)
    rfecv.fit(
        X, y, train_index_list, val_index_list, sample_weight_list=sample_weight_list
    )

    # organize dataframe for results
    n_gene = X.shape[1]
    cv_dict = rfecv.cv_results_.copy()
    # cv_dict.pop('mean_feature_ranking')
    cv_df = pd.DataFrame.from_dict(cv_dict)

    # find number of features selected in each iteration
    import math

    nfeat = n_gene
    step = step
    steps = [n_gene]
    while nfeat > 1:
        nstep = math.ceil(nfeat * step)
        nfeat = nfeat - nstep
        steps.append(nfeat)

    cv_df.index = steps[::-1]

    adata_train.uns["rfecv_result"] = cv_df
    adata_train.uns["rfecv_result_metric"] = rfecv.scoring

    if out_dir is not None:
        # save tmp output------------------
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        model_file = f"{out_dir}/rfecv_ranking_by_{type(model).__name__}.sav"
        pickle.dump(rfecv, open(model_file, "wb"))

    return adata_train, rfecv


def plot_gene_score(adata_train: AnnData, n_genes_plot: int=200, width: int=5, height: int=4, k: Optional[int]=None) -> Axes:

    cv_df = adata_train.uns["rfecv_result"].filter(regex="mean|split")
    cv_df = cv_df.loc[:n_genes_plot,]
    cv_df.columns = cv_df.columns.str.rstrip("_test_score")

    scoring_metrics = adata_train.uns["rfecv_result_metric"]
    if scoring_metrics == "average_precision":
        ylabel = "AUPRC"
    elif scoring_metrics == "roc_auc":
        ylabel = "AUROC"
    else:
        ylabel = scoring_metrics

    fig, axes = plt.subplots(figsize=(width, height))
    for columnName, columnData in cv_df.items():
        if "mean" in columnName:
            axes.plot(columnData, label=columnName)
        else:
            axes.plot(columnData, label=columnName, linestyle="dashed", alpha=0.6)

    axes.spines[["right", "top"]].set_visible(False)

    plt.xlabel("Number of Genes")
    plt.ylabel(ylabel)
    plt.legend()

    if k is not None:
        k_score = cv_df.loc[k, "mean"]
        y_label_adjust = (cv_df["mean"].max() - cv_df["mean"].min()) / 2

        plt.axvline(x=k, color="r", linestyle=":")
        plt.text(
            x=k + 4, y=k_score - y_label_adjust, s=f"n={k}\n{ylabel}={k_score:.3f}"
        )

    return axes


def decide_k(adata_train: AnnData, n_genes_plot: int=100) -> int:
    cv_df = adata_train.uns["rfecv_result"]
    cv_df = cv_df.loc[:n_genes_plot, :]

    data = cv_df.reset_index()[["index", "mean_test_score"]].to_numpy()
    A = data[0]
    B = data[-1]
    # 利用ABC三点坐标计算三角形面积，利用AB边长倒推三角形的高
    Dist = dict()
    for i in range(1, len(data)):
        C = data[i]
        ngene = C[0]
        D = np.append(np.vstack((A, B, C)), [[1], [1], [1]], axis=1)
        S = 1 / 2 * np.linalg.det(D)
        Dist[ngene] = 2 * S / np.linalg.norm(A - B)

    top_n_feat = int(max(Dist, key=Dist.get))
    top_n_feat_auc = cv_df.loc[max(Dist, key=Dist.get), "mean_test_score"]

    # print(f'Number of genes to select = {top_n_feat}')

    return top_n_feat


def select_gene(
    adata_train: AnnData, out_dir: Optional[str]=None, step: float=0.03, top_n_feat: int=5, n_genes_plot: int=100, verbose: int=0
) -> AnnData:

    # retrieve top_n_feat from one SVM-RFE run
    X, y = get_X_y_from_ann(adata_train)

    # Fill NaN in numpy
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    # model------------
    model = svm.SVC(kernel="linear", random_state=123)

    ## get ranking of all selected features
    selector = RFE(model, n_features_to_select=1, step=step, verbose=verbose)

    sample_weight = compute_cell_weight(adata_train)
    selector.fit(X, y, sample_weight=sample_weight)

    feature_ranking = pd.DataFrame(
        {"ranking": selector.ranking_}, index=adata_train.var_names
    ).sort_values(by="ranking")
    sig_list_ranked = feature_ranking.index[:top_n_feat].tolist()
    # print(sig_list_ranked)

    adata_train.uns["svm_rfe_genes"] = sig_list_ranked
    adata_train.var["ranking"] = selector.ranking_

    if out_dir is not None:
        # output gene list
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(f"{out_dir}/sig_svm.txt", "w") as f:
            for item in sig_list_ranked:
                f.write("%s\n" % item)

        # output adata_train_s with gene scores
        adata_train.write_h5ad(f"{out_dir}/adata_train_s.h5ad")

    return adata_train


def select_gene_stable(
    adata_train,
    n_iter=20,
    nfold=2,
    downsample_prop_list=[0.6, 0.8],
    num_cores=1,
    out_dir=None,
):

    def _single_fit(downsample_prop, i, adata_train, nfold, out_dir):

        downsample_size = round(adata_train.n_obs * downsample_prop)
        i = i + 1

        # create folder to output results for each iteration
        out_dir = f"{out_dir}/{downsample_size}/{i}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # metadata for stratified downsampling
        adata_train.obs["downsample_stratify"] = adata_train.obs[["patient_id"]].astype(
            "category"
        )

        down_index_i = resample(
            adata_train.obs_names,
            replace=False,
            n_samples=downsample_size,
            stratify=adata_train.obs["downsample_stratify"],
            random_state=i,
        )
        # downsampling
        adata_train_i = adata_train[adata_train.obs_names.isin(down_index_i),].copy()

        # QC for downsampled traninig data
        # 1. for each cell type, remove samples with <20 cells
        # 2. remove cell types with < 2 samples
        # 3. Remove 0-expressed genes
        # 4. Update training data

        min_cells = 20
        ## Number of cells in each patient
        n_cell_pt = adata_train_i.obs.groupby(
            ["patient_id"], observed=True, as_index=False
        ).size()
        # Remove paients with cells less than min_cells
        pt_keep = n_cell_pt.patient_id[n_cell_pt["size"] >= min_cells].tolist()

        ## Cell types with 0 patient has cells >= min_cells
        if len(pt_keep) > 0:
            adata_train_i = adata_train_i[
                adata_train_i.obs["patient_id"].isin(pt_keep),
            ]
            n_cell_pt = adata_train_i.obs.groupby(
                ["y", "patient_id"], observed=True, as_index=False
            ).size()
            ## Skip cell types with less than 2 patients in at least one condition
            if (n_cell_pt.y.nunique() >= 2) & ((n_cell_pt.y.value_counts() >= 2).all()):
                print("we have >= 2 samples in each condition...")

                ## Remove 0-expressed genes
                sc.pp.filter_genes(adata_train_i, min_cells=1)

                # Split downsampled train data into folds
                train_index_list, val_index_list, sample_weight_list = split_n_folds(
                    adata_train_i, nfold=nfold, out_dir=out_dir, random_state=2349
                )

                adata_train_i, rfecv_i = gene_score(
                    adata_train_i,
                    train_index_list,
                    val_index_list,
                    sample_weight_list=sample_weight_list,
                    step=0.03,
                    out_dir=out_dir,
                    ncpus=None,
                    verbose=False,
                )

                k = decide_k(adata_train_i, n_genes_plot=100)
                adata_train_i = select_gene(
                    adata_train_i, top_n_feat=k, step=0.03, out_dir=out_dir
                )
                sig_svm_i = adata_train_i.uns["svm_rfe_genes"]

        res_i = pd.DataFrame(sig_svm_i, columns=["gene"])
        res_i["downsample_prop"] = downsample_prop
        res_i["downsample_size"] = downsample_size
        res_i["n_iter"] = i

        return res_i

    start = time.time()

    paramlist = itertools.product(downsample_prop_list, range(n_iter))  # 2 nested loops
    res = Parallel(n_jobs=num_cores)(
        delayed(_single_fit)(
            downsample_prop, i, adata_train=adata_train, nfold=nfold, out_dir=out_dir
        )
        for downsample_prop, i in paramlist
    )
    end = time.time()

    res_df = pd.concat(res)
    gene_freq_df = res_df.groupby(
        ["downsample_prop", "downsample_size", "gene"], as_index=False
    ).size()
    adata_train.uns["scPanel_stable_rfecv_result"] = gene_freq_df

    gene_mean = gene_freq_df.groupby("gene")["size"].mean()
    gene_mean = gene_mean[gene_mean > round(n_iter * 0.5)].sort_values(ascending=False)
    adata_train.uns["svm_rfe_genes_stable"] = gene_mean.index.tolist()
    adata_train.uns["svm_rfe_genes_stable_time"] = end - start

    return adata_train
