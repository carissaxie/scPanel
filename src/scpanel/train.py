import os
import pickle
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# import sklearn.linear_model as lm
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .GATclassifier import GATclassifier

# import pandas as pd
# import numpy as np
from .utils_func import *
import sklearn.ensemble._forest
import sklearn.linear_model._logistic
import sklearn.neighbors._classification
import sklearn.svm._classes
from anndata._core.anndata import AnnData
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scpanel.GATclassifier import GATclassifier
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union


def transform_adata(adata_train: AnnData, adata_test_dict: Dict[str, AnnData], selected_gene: Optional[List[str]]=None) -> Tuple[AnnData, AnnData]:
    ## Transforming train set and test set from the same dataset (batch effect free)
    ## subset adata_train with selected genes
    ## subset adata_test_dict with selected cell types and genes
    ## WATCH OUT: X matrix in adata_test_dict is log-normalized, need to scale further
    if selected_gene == None:
        selected_gene = adata_train.uns["svm_rfe_genes"]

    adata_train_final = adata_train[:, selected_gene]

    mean = adata_train_final.var["mean"].values
    std = adata_train_final.var["std"].values

    ct_selected = adata_train_final.obs.ct.unique()[0]

    # transform test data with selected gene, celltype and scaling
    adata_test = adata_test_dict[ct_selected].copy()
    adata_test_final = adata_test[:, selected_gene].copy()

    if isinstance(adata_test_final.X, np.ndarray):
        test_X = adata_test_final.X
    else:
        test_X = adata_test_final.X.toarray()
    test_X -= mean
    test_X /= std

    max_value = 10
    test_X[test_X > max_value] = max_value
    adata_test_final.X = test_X

    return adata_train_final, adata_test_final


def models_train(adata_train_final: AnnData, search_grid: bool, out_dir: Optional[str]=None, param_grid: Optional[Dict[str, Dict[str, int]]]=None) -> List[Union[Tuple[str, sklearn.linear_model._logistic.LogisticRegression], Tuple[str, sklearn.ensemble._forest.RandomForestClassifier], Tuple[str, sklearn.svm._classes.SVC], Tuple[str, sklearn.neighbors._classification.KNeighborsClassifier], Tuple[str, GATclassifier]]]:

    X_tr, y_tr, adj_tr = get_X_y_from_ann(
        adata_train_final, return_adj=True, n_neigh=10
    )
    sample_weight = compute_cell_weight(adata_train_final)

    # Make sure no nan in matrix
    X_tr = np.nan_to_num(X_tr)

    grid_search = search_grid
    models = [
        ("LR", LogisticRegression(solver="saga", max_iter=500, random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("SVM", SVC(probability=True, random_state=42)),
        ("KNN", KNeighborsClassifier()),
        (
            "GAT",
            GATclassifier(
                nFeatures=adata_train_final.n_vars, NumParts=10, nEpochs=1000, verbose=1
            ),
        ),
    ]

    # Parameter tuning grids-------------------------
    LR_params = [{"C": [10, 1.0, 0.1, 0.01], "max_iter": [10, 50, 200, 500]}]
    RF_params = [
        {"max_depth": [2, 5, 10, 15, 20, 30, None], "n_estimators": [50, 100, 500]}
    ]
    SVM_params = [{"C": [100, 10, 1.0, 0.1, 0.001], "gamma": [1, 0.1, 0.01, 0.001]}]
    KNN_params = [{"n_neighbors": [3, 5, 10, 20, 50], "p": [1, 2]}]

    my_grid = {"LR": LR_params, "RF": RF_params, "SVM": SVM_params, "KNN": KNN_params}

    clfs = []
    names = []
    runtimes = []
    best_params = []

    for name, model in models:
        start_time = time.time()

        if grid_search:
            if name != "GAT":
                clf = GridSearchCV(
                    model, my_grid[name], cv=5, scoring="roc_auc", n_jobs=10
                )
            else:
                clf = model
        else:
            clf = model
            if param_grid is not None:
                if name in param_grid:
                    clf.set_params(**param_grid[name])

        if name == "GAT":
            clf.fit(X_tr, y_tr, adj_tr)
        elif name == "KNN":
            clf.fit(X_tr, y_tr)
        else:
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)

        runtime = time.time() - start_time

        # save outputs
        clfs.append((name, clf))
        names.append(name)
        runtimes.append(runtime)

        print("---%s finished in %s seconds ---" % (name, runtime))

    # save models
    if out_dir is not None:

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(f"{out_dir}/clfs.pkl", "wb") as f:
            pickle.dump(clfs, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

        with open(f"{out_dir}/adata_train_final.pkl", "wb") as f:
            pickle.dump(adata_train_final, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

    return clfs


def models_predict(clfs: List[Union[Tuple[str, sklearn.linear_model._logistic.LogisticRegression], Tuple[str, sklearn.ensemble._forest.RandomForestClassifier], Tuple[str, sklearn.svm._classes.SVC], Tuple[str, sklearn.neighbors._classification.KNeighborsClassifier], Tuple[str, GATclassifier]]], adata_test_final: AnnData, out_dir: Optional[str]=None) -> Tuple[AnnData, List[Union[Tuple[str, ndarray], Tuple[str, Tensor]]], List[Tuple[str, ndarray]]]:
    X_test, y_test, adj_test = get_X_y_from_ann(
        adata_test_final, return_adj=True, n_neigh=10
    )
    X_test = np.nan_to_num(X_test)

    ## Predicting---------------
    y_pred_list = []
    y_pred_score_list = []

    for name, clf in clfs:
        if name == "GAT":
            y_pred = clf.predict(X_test, y_test, adj_test)
            y_pred_score = clf.predict_proba(X_test, y_test, adj_test)
        else:
            y_pred = clf.predict(X_test)
            y_pred_score = clf.predict_proba(X_test)

        y_pred_list.append((name, y_pred))
        y_pred_score_list.append((name, y_pred_score))

    # add prediction result to adata_test_final
    y_pred = pd.DataFrame(dict([(name + "_pred", pred) for name, pred in y_pred_list]))
    y_pred_score = pd.DataFrame(
        dict([(name + "_pred_score", pred[:, 1]) for name, pred in y_pred_score_list])
    )

    y_pred_df = pd.concat([y_pred, y_pred_score], axis=1)
    y_pred_df.index = adata_test_final.obs.index

    if set(y_pred_df.columns).issubset(set(adata_test_final.obs.columns)):
        print("Prediction result already exits in test adata, overwrite it...")
        adata_test_final.obs.update(y_pred_df)
    else:
        adata_test_final.obs = pd.concat([adata_test_final.obs, y_pred_df], axis=1)

    # calcuate median prediction score out of 5 classifiers
    pred_col = [
        col for col in adata_test_final.obs.columns if col.endswith("_pred_score")
    ]
    adata_test_final.obs["median_pred_score"] = adata_test_final.obs[pred_col].median(
        axis=1
    )

    return adata_test_final, y_pred_list, y_pred_score_list


def models_score(adata_test_final, y_pred_list, y_pred_score_list, out_dir=None):
    X_test, y_test = get_X_y_from_ann(adata_test_final)

    ## Scoring-------------------------------------
    ## define scoring metrics (from sklearn)
    scorers = {
        "accuracy": (accuracy_score, {}),
        "balanced_accuracy": (balanced_accuracy_score, {}),
        "MCC": (matthews_corrcoef, {}),
    }  # Passing Dictionary as Arguments to Function

    scorers_prob = {
        "AUROC": (roc_auc_score, {}),
        "AUPRC": (average_precision_score, {}),
    }

    ## calculate
    eval_res_1 = pd.DataFrame()
    for name, y_pred in y_pred_list:
        eval_res_dict = dict(
            [
                (score_name, score_func(y_test, y_pred, **score_para))
                for score_name, (score_func, score_para) in scorers.items()
            ]
        )
        eval_res_i = pd.DataFrame(eval_res_dict, index=[name])

        eval_res_1 = pd.concat(objs=[eval_res_1, eval_res_i], axis=0)

    eval_res_2 = pd.DataFrame()
    for name, y_pred_score in y_pred_score_list:
        eval_res_dict = dict(
            [
                (score_name, score_func(y_test, y_pred_score[:, 1], **score_para))
                for score_name, (score_func, score_para) in scorers_prob.items()
            ]
        )
        eval_res_i = pd.DataFrame(eval_res_dict, index=[name])

        eval_res_2 = pd.concat(objs=[eval_res_2, eval_res_i], axis=0)

    eval_res = pd.concat(objs=[eval_res_2, eval_res_1], axis=1)

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        eval_res.to_csv(f"{out_dir}/eval_res.csv")

    return eval_res


def cal_sample_auc(df: DataFrame, score_col: str) -> float64:
    cell_prob = df[score_col].sort_values()
    # rank the cell probability ascendingly and normalize
    cell_rank = cell_prob.rank(method="first") / cell_prob.rank(method="first").max()
    sample_auc = auc(cell_rank, cell_prob)
    return sample_auc


def auc_pvalue(row: Series) -> float:
    if row.name[1] == 1:
        p_value = np.mean(row < 0.5)
    elif row.name[1] == 0:
        p_value = np.mean(row > 0.5)

    if p_value == 0:
        p_value = 1 / row.size
    return p_value


def pt_pred(adata_test_final: AnnData, cell_pred_col: str="median_pred_score", num_bootstrap: Optional[int]=None) -> AnnData:
    sample_auc = adata_test_final.obs.groupby("patient_id").apply(
        lambda df: cal_sample_auc(df, cell_pred_col)
    )
    adata_test_final.obs[cell_pred_col + "_sample_auc"] = (
        adata_test_final.obs["patient_id"].map(sample_auc).astype(float)
    )
    adata_test_final.obs[cell_pred_col + "_sample_pred"] = (
        adata_test_final.obs[cell_pred_col + "_sample_auc"] >= 0.5
    ).astype(int)

    if num_bootstrap is not None:
        auc_df = pd.DataFrame()
        for i in range(num_bootstrap):
            df = adata_test_final.obs.groupby("patient_id").sample(
                frac=1, replace=True, random_state=i
            )
            auc = (
                df.groupby(["patient_id", cell_pred_col + "_sample_pred"])
                .apply(lambda df: cal_sample_auc(df, cell_pred_col))
                .to_frame(name=i)
            )
            auc_df = pd.concat([auc_df, auc], axis=1)

        auc_df[cell_pred_col + "_sample_auc_pvalue"] = auc_df.apply(
            lambda row: auc_pvalue(row), axis=1
        )
        # store auc from each bootstrap iteration in adata.uns
        adata_test_final.uns[cell_pred_col + "_auc_df"] = auc_df
        # store auc_pvalue for each sample in adata.obs
        auc_df = auc_df.droplevel(cell_pred_col + "_sample_pred")
        adata_test_final.obs[cell_pred_col + "_sample_auc_pvalue"] = (
            adata_test_final.obs["patient_id"].map(
                auc_df[cell_pred_col + "_sample_auc_pvalue"]
            )
        )

    return adata_test_final


def pt_score(adata_test_final: AnnData, cell_pred_col: str="median_pred_score") -> AnnData:
    ## Calculate precision, recall, f1score and accuracy at patient level
    from sklearn.metrics import precision_recall_fscore_support

    pred_col = cell_pred_col
    res_prefix = cell_pred_col

    pt_pred_res = (
        adata_test_final.obs[["label", "patient_id", f"{res_prefix}_sample_pred"]]
        .drop_duplicates()
        .set_index("patient_id")
    )

    # precision, recall, f1score
    pt_score_res = precision_recall_fscore_support(
        pt_pred_res["label"],
        pt_pred_res[f"{res_prefix}_sample_pred"],
        average="weighted",
    )
    # accuracy
    pt_acc_res = accuracy_score(
        pt_pred_res["label"], pt_pred_res[f"{res_prefix}_sample_pred"]
    )
    # specificity
    pt_spec_res = recall_score(
        pt_pred_res["label"], pt_pred_res[f"{res_prefix}_sample_pred"], pos_label=0
    )

    pt_score_res = pd.DataFrame(list(pt_score_res) + [pt_acc_res] + [pt_spec_res])
    pt_score_res = pt_score_res.iloc[[0, 1, 2, 4, 5], :]
    pt_score_res.index = [
        "precision",
        "sensitivity",
        "f1score",
        "accuracy",
        "specificity",
    ]
    pt_score_res.columns = [res_prefix]

    if "sample_score" not in adata_test_final.uns:
        adata_test_final.uns["sample_score"] = pt_score_res
    else:
        adata_test_final.uns["sample_score"] = adata_test_final.uns[
            "sample_score"
        ].merge(pt_score_res, left_index=True, right_index=True, suffixes=("_x", ""))

        adata_test_final.uns["sample_score"].drop(
            adata_test_final.uns["sample_score"].filter(regex="_x$").columns,
            axis=1,
            inplace=True,
        )

    return adata_test_final


from math import pi

# Plot functions
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def _panel_grid(hspace: float, wspace: float, ncols: int, num_panels: int) -> Tuple[Figure, GridSpec]:
    from matplotlib import gridspec

    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    # each panel will have the size of rcParams['figure.figsize']
    fig = plt.figure(
        figsize=(
            n_panels_x * rcParams["figure.figsize"][0] * (1 + wspace),
            n_panels_y * rcParams["figure.figsize"][1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, gs


def plot_roc_curve(
    adata_test_final: AnnData,
    sample_id: Series,
    cell_pred_col: str,
    ncols: int=4,
    hspace: float=0.25,
    wspace: None=None,
    ax: None=None,
    scatter_kws: Optional[Dict[str, int]]=None,
    legend_kws: Optional[Dict[str, Dict[str, int]]]=None,
) -> List[Axes]:
    """
    Parameters
    ----------
    - adata_test_final: AnnData,
    - sample_id: str | Sequence,
    - cell_pred_col: str = 'median_pred_score',
    - ncols: int = 4,
    - hspace: float =0.25,
    - wspace: float | None = None,
    - ax: Axes | None = None,
    - scatter_kws: dict | None = None, Arguments to pass to matplotlib.pyplot.scatter()

    Returns
    -------
    Axes

    Examples
    --------
    plot_roc_curve(adata_test_final,
               sample_id = ['C3','C6','H1'],
               cell_pred_col = 'median_pred_score',
               scatter_kws={'s':10})

    """

    # turn sample_id into a python list
    ## if sample_id is string or None, wrap it with []
    ## if sample_id is already sequential, turn it into a list
    sample_id = (
        [sample_id]
        if isinstance(sample_id, str) or sample_id is None
        else list(sample_id)
    )

    ##########
    # Layout #
    ##########
    if scatter_kws is None:
        scatter_kws = {}

    if legend_kws is None:
        legend_kws = {}

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02

    # if plotting multiple panels for elements in sample_id
    if len(sample_id) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )
        fig, grid = _panel_grid(hspace, wspace, ncols, len(sample_id))
    else:
        grid = None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

    ############
    # Plotting #
    ############
    axs = []
    for count, _sample_id in enumerate(sample_id):
        if grid:
            ax = plt.subplot(grid[count])
            axs.append(ax)

        # prediction probability of class 1 for sample_id
        cell_prob = adata_test_final.obs.loc[
            adata_test_final.obs["patient_id"] == sample_id[count]
        ][cell_pred_col]
        cell_prob = cell_prob.sort_values(ascending=True)
        # rank of cell_prob and normalize
        cell_rank = (
            cell_prob.rank(method="first") / cell_prob.rank(method="first").max()
        )
        # auc
        sample_auc = adata_test_final.obs.loc[
            adata_test_final.obs["patient_id"] == sample_id[count]
        ][cell_pred_col + "_sample_auc"].unique()[0]
        # auc-pvalue
        sample_auc_pvalue = adata_test_final.obs.loc[
            adata_test_final.obs["patient_id"] == sample_id[count]
        ][cell_pred_col + "_sample_auc_pvalue"].unique()[0]

        ax.scatter(x=cell_rank, y=cell_prob, c=".3", **scatter_kws)
        ax.plot(
            cell_rank,
            cell_prob,
            label=f"AUC = {sample_auc:.3f} \np-value = {sample_auc_pvalue:.1e}",
            zorder=0,
        )
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color=".5", zorder=0, label="Random guess"
        )
        # ax.text(x = 0.99, y = 0.01, s = f'AUC: {sample_auc:.3f}',
        #         horizontalalignment='right',
        #         verticalalignment='bottom')
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Prediction Probability (Cell)")
        ax.set_title(f"{_sample_id}")
        ax.set_aspect("equal")
        if not bool(legend_kws):
            ax.legend(prop=dict(size=8 * rcParams["figure.figsize"][0] / ncols))
        else:
            ax.legend(**legend_kws)

    axs = axs if grid else ax

    return axs


def convert_pvalue_to_asterisks(pvalue: float) -> str:
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


# plot cell level probabilities for each patient
def plot_violin(
    adata: AnnData,
    cell_pred_col: str="median_pred_score",
    dot_size: int=2,
    ax: Optional[Axes]=None,
    palette: Optional[Dict[str, str]]=None,
    xticklabels_color: bool=False,
    text_kws: Dict[Any, Any]={},
) -> Axes:
    """
    Violin Plots for cell-level prediction probabilities in each sample.

    Parameters:
    - adata: AnnData Object

    - cell_pred_col: string, name of the column with cell-level prediction probabilities
    in adata.obs (default: 'median_pred_score')

    - pt_stat: string, a test for the null hypothesis that the distribution of probabilities
    in this sample is different from the population (default: 'perm')
        Options:
        - 'perm': permutation test
        - 't-test': one-sample t-test

    - fig_size: tuple, size of figure (default: (10, 3))
    - dot_size: float, Radius of the markers in stripplot.

    Returns:
        ax

    """

    # A. organize input data for plotting--------------
    res_prefix = cell_pred_col
    ## cell-level data
    pred_score_df = adata.obs[
        [
            cell_pred_col,
            "y",
            "label",
            "patient_id",
            f"{res_prefix}_sample_auc",
            f"{res_prefix}_sample_auc_pvalue",
        ]
    ].copy()

    ## sample-level data
    sample_pData = pred_score_df.groupby(
        [
            "y",
            "label",
            "patient_id",
            f"{res_prefix}_sample_auc",
            f"{res_prefix}_sample_auc_pvalue",
        ],
        observed=True,
        as_index=False,
    )[cell_pred_col].max()
    sample_pData.rename(columns={cell_pred_col: "y_pos"}, inplace=True)
    sample_pData = sample_pData.sort_values(by=f"{res_prefix}_sample_auc").reset_index(
        drop=True
    )

    sample_order = sample_pData.patient_id.tolist()

    sample_threshold_index = (
        sample_pData[f"{res_prefix}_sample_auc"]
        .where(sample_pData[f"{res_prefix}_sample_auc"] >= 0.5)
        .first_valid_index()
    )
    if sample_threshold_index is None:
        if (sample_pData[f"{res_prefix}_sample_auc"] >= 0.5).all():
            sample_threshold = -0.5
        else:
            sample_threshold = len(sample_pData[f"{res_prefix}_sample_auc"]) - 0.5
    else:
        sample_threshold = sample_threshold_index - 0.5

    # B. plot--------------------------------------------
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines[["right", "top"]].set_visible(False)

    # Violin plot
    sns.violinplot(
        y=cell_pred_col,
        x="patient_id",
        data=pred_score_df,
        order=sample_order,
        color="0.8",
        scale="width",
        fontsize=15,
        ax=ax,
        cut=0,
    )

    # Strip plot
    sns.stripplot(
        y=cell_pred_col,
        x="patient_id",
        hue="y",
        data=pred_score_df,
        order=sample_order,
        dodge=False,
        jitter=True,
        size=dot_size,
        ax=ax,
        palette=palette,
    )

    ax.axhline(y=0.5, color="0.8", linestyle="--")
    ax.axvline(x=sample_threshold, color="0.8", linestyle="--")

    # Add statistical signifiance (asterisks (*)) on top of each violin
    ## get position x
    yposlist = (sample_pData["y_pos"] + 0.03).tolist()
    ## get position y
    xposlist = range(len(yposlist))
    ## get text
    pvalue_list = sample_pData[f"{res_prefix}_sample_auc_pvalue"].tolist()
    asterisks_list = [convert_pvalue_to_asterisks(pvalue) for pvalue in pvalue_list]
    perm_stat_list = [
        "%.3f" % perm_stat
        for perm_stat in sample_pData[f"{res_prefix}_sample_auc"].tolist()
    ]
    text_list = [
        perm_stat + "\n" + asterisk
        for perm_stat, asterisk in zip(perm_stat_list, asterisks_list)
    ]

    for k in range(len(asterisks_list)):
        ax.text(x=xposlist[k], y=yposlist[k], s=text_list[k], ha="center", **text_kws)

    ax.set_title(cell_pred_col, pad=30)
    ax.set_xlabel(None)
    ax.set_ylabel("Prediction Probablity (Cell)", fontsize=13)
    ax.plot()

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    if xticklabels_color:
        for xtick in ax.get_xticklabels():
            x_label = xtick.get_text()
            x_label_cate = sample_pData["y"][
                sample_pData["patient_id"] == x_label
            ].values[0]
            xtick.set_color(palette[x_label_cate])

    ax.legend(loc="upper left", title="Patient Label", bbox_to_anchor=(1.04, 1))

    return ax


### Plot patient level prediction scores
def make_single_spider(adata_test_final: AnnData, metric_idx: int, color: str, nrow: int, ncol: int) -> None:
    # number of variable
    categories = adata_test_final.uns["sample_score"].index.tolist()
    N = len(adata_test_final.uns["sample_score"].index)

    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = (
        adata_test_final.uns["sample_score"]
        .iloc[:, metric_idx]
        .values.flatten()
        .tolist()
    )
    values += values[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(nrow, ncol, metric_idx + 1, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color="grey", size=15)

    for label, i in zip(ax.get_xticklabels(), range(0, len(angles))):
        if i < len(angles) / 2:
            angle_text = angles[i] * (-180 / pi) + 90
            label.set_horizontalalignment("left")

        else:
            angle_text = angles[i] * (-180 / pi) - 90
            label.set_horizontalalignment("right")
        label.set_rotation(angle_text)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.6], ["0.1", "0.3", "0.6"], color="grey", size=8)
    plt.ylim(0, 1.05)

    # Plot data
    ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.4)
    ax.grid(color="white")
    for ti, di in zip(angles, values):
        ax.text(
            ti, di - 0.02, "{0:.2f}".format(di), color="black", ha="center", va="center"
        )

    # Add a title
    t = adata_test_final.uns["sample_score"].columns[metric_idx]
    t = t.replace("_pred_score", "")
    plt.title(t, color="black", y=1.2, size=22)
