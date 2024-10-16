# import scanpy as sc

import os

# import numpy as np
# import pandas as pd
import time
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier

# from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils_func import *
from anndata._core.anndata import AnnData
from matplotlib.axes._axes import Axes
from numpy import float64
from pandas.core.frame import DataFrame
from typing import Dict, Tuple


def split_patients(adata: AnnData, test_pt_size: float, random_state: int, out_dir: str, verbose: bool) -> Tuple[AnnData, AnnData]:
    #######################################################
    # utilize anndata (based on Scanpy) as input and output
    ## - adata: preprocessed, X contains log-transformed data
    ## - random_state
    ## - out_dir: save splitted data and split information
    ## - dataset: name of dataset
    #######################################################

    # retrieve patient-level and cell-level metadata
    pat_meta_temp = adata.obs[["y", "patient_id"]].drop_duplicates().reset_index()
    cell_meta_temp = adata.obs

    # split train and test set at patient-level
    rest_, patient_test_id_list = train_test_split(
        pat_meta_temp.patient_id,
        test_size=test_pt_size,
        stratify=pat_meta_temp.y,
        random_state=random_state,
    )
    if verbose:
        print(
            len(patient_test_id_list),
            "patients in test set: ",
            patient_test_id_list.values,
        )
        print(len(rest_), "patients in train set: ", rest_.values)

    # retrieve cell-level index for train and test set
    test_idx = cell_meta_temp[
        cell_meta_temp["patient_id"].isin(patient_test_id_list)
    ].index.tolist()
    train_idx = cell_meta_temp[cell_meta_temp["patient_id"].isin(rest_)].index.tolist()

    ## reload train-test index-----------------
    ## is_train = np.genfromtxt(f'{out_dir}/train_test_idx.txt', dtype=bool)

    # retreive y (labels)
    y = adata.obs["y"]

    # retreive X (data)
    train_adata = adata[train_idx, :]
    test_adata = adata[test_idx, :]

    # output split information
    ## train set
    cell_info_train = pd.DataFrame(
        dict(
            **Counter(y[train_idx]),
            **{"total_cells": len(train_idx), "patient_ids": [rest_.values]},
        ),
        index=["train"],
    )
    n_patient_train = (
        train_adata.obs[["y", "patient_id"]]
        .drop_duplicates()
        .groupby(["y"])["patient_id"]
        .count()
        .to_frame()
        .T
    )
    n_patient_train.columns = ["N_" + x for x in n_patient_train.columns.tolist()]
    split_info_train = pd.concat(
        [cell_info_train, n_patient_train.set_index(cell_info_train.index)], axis=1
    )

    ## test set
    cell_info_test = pd.DataFrame(
        dict(
            **Counter(y[test_idx]),
            **{
                "total_cells": len(test_idx),
                "patient_ids": [patient_test_id_list.values],
            },
        ),
        index=["test"],
    )
    n_patient_test = (
        test_adata.obs[["y", "patient_id"]]
        .drop_duplicates()
        .groupby(["y"])["patient_id"]
        .count()
        .to_frame()
        .T
    )
    n_patient_test.columns = ["N_" + x for x in n_patient_test.columns.tolist()]
    split_info_test = pd.concat(
        [cell_info_test, n_patient_test.set_index(cell_info_test.index)], axis=1
    )

    split_info = pd.concat([split_info_train, split_info_test], axis=0)
    # print(split_info)

    # output
    ## split information

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # output train and test index
    np.savetxt(
        f"{out_dir}/train_test_idx.txt", adata.obs_names.isin(train_idx), fmt="%s"
    )
    split_info.to_csv(f"{out_dir}/split_patient_train_test_info.csv")

    ## train set & test set
    # del train_adata.raw
    # del test_adata.raw
    train_adata.write_h5ad(f"{out_dir}/processed_rna_assay_train.h5ad")
    test_adata.write_h5ad(f"{out_dir}/processed_rna_assay_test.h5ad")

    return train_adata, test_adata


def cal_bootstrap_score(
    adata: AnnData,
    out_dir: str,
    sample_n_cell: int,
    n_iterations: int=100,
    n_threads: int=16,
    show_progress: bool=True,
    verbose: bool=False,
) -> DataFrame:

    celltype = adata.obs["ct"].unique()[0]

    # Initializing DataFrame, to hold bootstrapped statistics
    bootstrapped_stats = pd.DataFrame()

    if show_progress:
        print(celltype, "start calculating...")
    # Each loop iteration is a single bootstrap resample and model fit
    for i in range(n_iterations):

        if verbose:
            print("Starting iteration #", i)

        adata_0 = adata[adata.obs.label == 0].copy()
        adata_1 = adata[adata.obs.label == 1].copy()

        # make balanced data
        adata_0_i_index = (
            adata_0.obs.groupby("patient_id")
            .sample(n=sample_n_cell, replace=False, random_state=i)
            .index
        )
        adata_0_i = adata_0[adata_0_i_index]
        adata_1_i_index = (
            adata_1.obs.groupby("patient_id")
            .sample(n=sample_n_cell, replace=False, random_state=i)
            .index
        )
        adata_1_i = adata_1[adata_1_i_index]

        adata_i = anndata.concat([adata_0_i, adata_1_i])
        adata_i.obs_names_make_unique()

        adata_train_i, adata_test_i = split_patients(
            adata_i,
            random_state=i,
            test_pt_size=0.4,
            out_dir=f"{out_dir}/tmp/split_{i}",
            verbose=verbose,
        )

        X_i_train, y_i_train = get_X_y_from_ann(adata_train_i)
        X_i_test, y_i_test = get_X_y_from_ann(adata_test_i)

        # Fill NaN in numpy
        X_i_train = np.nan_to_num(X_i_train)
        y_i_train = np.nan_to_num(y_i_train)
        X_i_test = np.nan_to_num(X_i_test)
        y_i_test = np.nan_to_num(y_i_test)

        # Initializing estimator
        rf = RandomForestClassifier(
            n_jobs=n_threads, class_weight="balanced", random_state=i
        )
        rf.fit(X_i_train, y_i_train)

        # Make prediction
        y_i_pred = rf.predict(X_i_test)
        y_i_pred_score = rf.predict_proba(X_i_test)

        # Storing stats in DataFrame, and concatenating with stats
        bACC = balanced_accuracy_score(y_i_test, y_i_pred)
        # AUC = roc_auc_score(y_i_test, y_i_pred_score[:, 1])
        AUC = roc_auc_score(y_i_test, y_i_pred_score[:, 1])

        bootstrapped_stats_i = pd.DataFrame(
            data=dict(bACC=bACC, AUC=AUC, celltype=celltype), index=[i]
        )

        bootstrapped_stats = pd.concat(objs=[bootstrapped_stats, bootstrapped_stats_i])

        if show_progress & ((i + 1) % 10 == 0):
            print("n_iterations", i + 1, " is done")

    return bootstrapped_stats


def custom_metrics(grouping: Tuple[str, DataFrame], metric: str) -> Dict[str, float64]:
    (group_label, df) = grouping

    if shapiro(df[metric])[1] <= 0.05:
        return {group_label: df[metric].median()}
    else:
        return {group_label: df[metric].mean()}


def cell_type_score(
    adata_train_dict: Dict[str, AnnData], out_dir: str, ncpus: int, sample_n_cell: int, n_iterations: int=100, verbose: bool=False
) -> Tuple[DataFrame, DataFrame]:

    bootstrapped_stats_all = pd.DataFrame()
    celltypes_all = adata_train_dict.keys()

    # timestart = time.time()
    for celltype in tqdm(celltypes_all):

        adata = adata_train_dict[celltype]

        bootstrapped_stats_celltype = cal_bootstrap_score(
            adata,
            n_iterations=n_iterations,
            sample_n_cell=sample_n_cell,
            n_threads=ncpus,
            show_progress=False,
            out_dir=out_dir,
            verbose=verbose,
        )

        bootstrapped_stats_all = pd.concat(
            objs=[bootstrapped_stats_all, bootstrapped_stats_celltype]
        )
        print(celltype, " DONE")

    timeend = time.time()
    # print ("Cell type scores calculation took", time.strftime('%Hh%Mm%Ss',time.gmtime(timeend - timestart)))

    grouping = bootstrapped_stats_all.groupby("celltype")

    AUC_dict = dict()
    for i in grouping:
        AUC_dict.update(custom_metrics(i, "AUC"))

    result_AUC = pd.DataFrame.from_dict(AUC_dict, orient="index", columns=["AUC"])

    # add number of cells in each cell type as one column
    n_cell_df = pd.DataFrame.from_dict(
        (dict((k, len(v)) for k, v in adata_train_dict.items())),
        orient="index",
        columns=["n_cell"],
    )
    result_AUC = result_AUC.merge(n_cell_df, left_index=True, right_index=True)
    result_AUC.index = result_AUC.index.set_names(["celltype"])
    result_AUC = result_AUC.reset_index().sort_values(by=["AUC"])

    # check if output path exist
    # if not, create one
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    result_AUC.to_csv(f"{out_dir}/celltype_AUC.csv")

    bootstrapped_stats_all.to_csv(f"{out_dir}/celltype_bootstrap_stats_all.csv")

    return result_AUC, bootstrapped_stats_all


def plot_cell_type_score(AUC: DataFrame, AUC_all: DataFrame, width: int=4, height: int=5) -> Axes:

    AUC = AUC.set_index("celltype")

    # long to wide table, sorted
    pData = AUC_all.pivot(columns="celltype", values="AUC")
    pData = pData[AUC.index[::-1]]

    # Initialize figure
    fig, axes = plt.subplots(figsize=(width, height), dpi=200)

    axes = sns.boxplot(data=pData, orient="h", ax=axes)
    axes.set_xlabel("AUROC")
    axes.set_ylabel("Disease Responsive Cell Types")

    # customize ytick labels
    axes.set_yticklabels(
        [y + " (" + str(AUC.loc[y, "n_cell"]) + ")" for y in pData.columns]
    )

    # customize xtick labels
    axes.set_xlim([0.5, 1.05])

    axes.spines[["right", "top"]].set_visible(False)

    # Add median value on top of each box
    yticks_dict = {k: v for k, v in zip(pData.columns, plt.yticks()[0])}
    for y, x in pData.max().items():
        s = AUC.loc[y, "AUC"]
        plt.text(
            x + 0.1,
            yticks_dict[y],
            f"{s:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
        )
    return axes


####deprecated 2023.11.17###########
# def plot_cell_type_score(result_AUC, width=5, height=5):

#     plot_df = result_AUC
#     plot_df['yname'] = plot_df.celltype + " (" + plot_df.n_cell.map(str) + ")"

#     # using subplots() to draw vertical lines
#     fig, axes = plt.subplots(figsize=(width, height), dpi=200)

#     # providing list of colors

#     axes.hlines(plot_df['yname'], xmin=0,
#                 xmax=plot_df['AUC'])

#     # drawing the markers (circle)
#     axes.plot(plot_df['AUC'], plot_df['yname'], "o")
#     axes.set_xlim(0)

#     # formatting and details
#     plt.xlabel('AUROC',fontsize=15)
#     plt.ylabel('Disease Responsive Cell Types',fontsize=15)
#     #plt.title('AUC')
#     plt.yticks(plot_df['yname'],fontsize=12)
#     plt.xlim([0, 1.2])

#     # expand the xlim but hide the last xtick '1.2'
#     x_ticks = axes.xaxis.get_major_ticks()
#     x_ticks[-1].set_visible(False)

#     yticks_dict = {k: v for k, v in zip(plot_df['celltype'], plt.yticks()[0])}
#     for x, y in zip(plot_df['AUC'], plot_df['celltype']):
#         plt.text(x+0.1, yticks_dict[y], round(x, 3), horizontalalignment='center', verticalalignment='center',
#                 fontsize=12)


def select_celltype(adata_train_dict: Dict[str, AnnData], out_dir: str, celltype_selected: str) -> AnnData:
    # output selected cell type
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(f"{out_dir}/selected_celltype.txt", "w") as f:
        for item in celltype_selected:
            f.write("%s" % item)

    adata_train = adata_train_dict[celltype_selected]

    sc.pp.filter_genes(adata_train, min_cells=1)

    print("Selecting ", *celltype_selected, "...")
    return adata_train.copy()
