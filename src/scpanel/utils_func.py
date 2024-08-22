import warnings

# from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.utils import resample
from anndata._core.anndata import AnnData
from numpy import ndarray
from scipy.sparse._base import spmatrix
from pandas.core.arrays.categorical import Categorical
from pandas.core.frame import DataFrame
from scipy.sparse._csr import csr_matrix

warnings.filterwarnings("ignore")


def preprocess(
    adata: object,
    integrated: bool=False,
    ct_col: Optional[str]=None,
    y_col: Optional[str]=None,
    pt_col: Optional[str]=None,
    class_map: Optional[Dict[str, int]]=None,
) -> AnnData:
    """standardize input data

    Parameters
    ----------
    adata : object
    integrated : bool=False
    ct_col : Optional[str]=None
    y_col : Optional[str]=None
    pt_col : Optional[str]=None
    class_map : Optional[Dict[str, int]]=None

    Returns
    -------
    AnnData

    """
    try:
        adata.__dict__["_raw"].__dict__["_var"] = (
            adata.__dict__["_raw"]
            .__dict__["_var"]
            .rename(columns={"_index": "features"})
        )
    except AttributeError as e:
        # print(f"An AttributeError occurred: {e}")
        # Handle the error or do nothing if you just want to continue
        pass

    # adata.raw = adata
    # can be recoverd by: adata_raw = adata.raw.to_adata()

    if not integrated:
        # Standardize X matrix-----------------------------------------
        ## 1. Check if X is raw count matrix
        ### If True, make X into logcount matrix
        if check_nonnegative_integers(adata.X):
            adata.layers["count"] = adata.X
            print("X is raw count matrix, adding to `count` layer......")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        elif check_nonnegative_float(adata.X):
            adata.layers["logcount"] = adata.X
            print("X is logcount matrix, adding to `logcount` layer......")
        else:
            raise TypeError("X should be either raw counts or log-normalized counts")

    # Filter low-express genes-----------------------------------------------
    sc.pp.filter_genes(adata, min_cells=1)

    # Calculate HVGs-------------------------------------------------------
    # sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0)

    # Standardize Metadata-----------------------------------------
    ## cell type as ct
    ## coondition as y

    if ct_col is not None:
        adata.obs["ct"] = adata.obs[ct_col]
        adata.obs["ct"] = adata.obs["ct"].astype("category")
    else:
        adata.obs["ct"] = adata.obs["ct"].astype("category")

    if y_col is not None:
        adata.obs["y"] = adata.obs[y_col]
        adata.obs["y"] = adata.obs["y"].astype("category")
    else:
        adata.obs["y"] = adata.obs["y"].astype("category")

    if pt_col is not None:
        adata.obs["patient_id"] = adata.obs[pt_col]
        adata.obs["patient_id"] = adata.obs["patient_id"].astype("category")
    else:
        adata.obs["patient_id"] = adata.obs["patient_id"].astype("category")

    # print(class_map)
    adata.obs["label"] = adata.obs["y"].map(class_map)

    #     # add HVGs in each cell type
    #     # split each cell type into new objects.
    #     celltypes_all = adata.obs['ct'].cat.categories.tolist()

    #     alldata = {}
    #     for celltype in celltypes_all:
    #         #print(celltype)
    #         adata_tmp = adata[adata.obs['ct'] == celltype,]
    #         if adata_tmp.shape[0] > 20:
    #             sc.pp.highly_variable_genes(adata_tmp, min_mean=0.0125, max_mean=3, min_disp=1.5)
    #             adata_tmp = adata_tmp[:, adata_tmp.var.highly_variable]
    #             alldata[celltype] = adata_tmp

    #     celltype_hvgs = anndata.concat(alldata, join="outer").var_names.tolist()
    #     hvgs = adata.var.highly_variable.index[adata.var.highly_variable].tolist()
    #     hvgs_idx = adata.var_names.isin(list(set(celltype_hvgs + hvgs)))

    return adata


def get_X_y_from_ann(adata: AnnData, return_adj: bool=False, n_neigh: int=10) -> Union[Tuple[ndarray, Categorical], Tuple[ndarray, Categorical, csr_matrix]]:
    """

    Parameters
    ----------
    adata
    return_adj
    n_neigh

    Returns
    -------

    """
    # scale and store results in layer
    # adata.layers['scaled'] = sc.pp.scale(adata, max_value=10, copy=True).X

    # This X matrix should be scaled value
    X = adata.to_df().values

    y = adata.obs["label"].values

    if return_adj:
        if adata.n_vars >= 50:
            sc.pp.pca(adata, n_comps=30, use_highly_variable=False)
            sc.pl.pca_variance_ratio(adata, log=True, n_pcs=30)
            knn = sc.Neighbors(adata)
            knn.compute_neighbors(n_neighbors=n_neigh, n_pcs=30)
        else:
            print(
                "Number of input genes < 50, use original space to get adjacency matrix..."
            )
            knn = sc.Neighbors(adata)
            knn.compute_neighbors(n_neighbors=n_neigh, n_pcs=0)

        adj = knn.connectivities + sparse.diags([1] * adata.shape[0]).tocsr()

        return X, y, adj

    else:
        return X, y


def check_nonnegative_integers(X: Union[np.ndarray, sparse.spmatrix]) -> bool:
    """Checks values of X to ensure it is count data"""
    from numbers import Integral

    data = X[np.random.choice(X.shape[0], 10, replace=False), :]
    data = data.todense() if sparse.issparse(data) else data

    # Check no negatives
    if np.signbit(data).any():
        return False
    # Check all are integers
    elif issubclass(data.dtype.type, Integral):
        return True
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True


def check_nonnegative_float(X: Union[np.ndarray, sparse.spmatrix]) -> bool:
    """Checks values of X to ensure it is logcount data"""
    from numbers import Integral

    data = X[np.random.choice(X.shape[0], 10, replace=False), :]
    data = data.todense() if sparse.issparse(data) else data

    # Check no negatives
    if np.signbit(data).any():  # one negative(if True), return False
        return False  # non negative(if False), go to next condition
    # Check all are integers
    elif np.issubdtype(data.dtype, np.floating):
        return True


def compute_cell_weight(data: Union[AnnData, DataFrame]) -> ndarray:
    ## data: anndata object | DataFrame from anndata.obs
    if isinstance(data, anndata.AnnData):
        data = data.obs

    from sklearn.utils.class_weight import compute_sample_weight

    ## class weight
    w_c = compute_sample_weight(class_weight="balanced", y=data["label"])
    w_c_df = pd.DataFrame({"patient_id": data["patient_id"], "w_c": w_c})

    ## patient weight
    gped = data.groupby(["label"])
    w_pt_df = pd.DataFrame([])
    for name, group in gped:
        w_pt = compute_sample_weight(class_weight="balanced", y=group["patient_id"])
        w_pt_df = pd.concat(
            [
                w_pt_df,
                pd.DataFrame(
                    {
                        "label": group["label"],
                        "patient_id": group["patient_id"],
                        "w_pt": w_pt,
                    }
                ),
            ],
            axis=0,
        )

    ## cell_weight = class weight*patient_weight
    w_df = w_c_df.merge(w_pt_df, left_index=True, right_index=True)
    w_df["w"] = w_df["w_c"] * w_df["w_pt"]

    ## Add results to adata metadata or save csv???
    # adata_train.obs = adata_train.obs.merge(w_df[['w_pt','w_c','w']], left_index=True, right_index=True)

    return w_df["w"].values


def downsample_adata(adata, downsample_size=4000, random_state=1):
    #######################
    ## stratified downsampling by patient_id, y, and ct (patient, disease status, and cell type)
    # metadata for stratified downsampling
    adata.obs["downsample_stratify"] = adata.obs[["patient_id", "y", "ct"]].apply(
        lambda x: "_".join(x.dropna().astype(str)), axis=1
    )
    adata.obs["downsample_stratify"] = adata.obs["downsample_stratify"].astype(
        "category"
    )

    down_index = resample(
        adata.obs_names,
        replace=False,
        n_samples=downsample_size,
        stratify=adata.obs["downsample_stratify"],
        random_state=random_state,
    )

    adata = adata[down_index,]
    return adata
