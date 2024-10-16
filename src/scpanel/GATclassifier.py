import copy
import inspect
import os.path as osp
import random

# import os,sys,pickle,time,random,glob
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.base import BaseEstimator
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor, cat
import torch_geometric.data.data
from numpy import ndarray
from pandas.core.arrays.categorical import Categorical
from scipy.sparse._csr import csr_matrix

# from .utils_func import get_X_y_from_ann


# import pandas as pd


# Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def scipysparse2torchsparse(x: csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: scipy csr_matrix
    Returns: torch tensor in experimental sparse format

    REF: Code adatped from [PyTorch discussion forum](https://discuss.pytorch.org/t/better-way-to-forward-sparse-matrix/21915>)
    """
    samples = x.shape[0]
    features = x.shape[1]
    values = x.data
    coo_data = x.tocoo()
    indices = torch.LongTensor(
        [coo_data.row, coo_data.col]
    )  # OR transpose list of index tuples
    t = torch.sparse.FloatTensor(
        indices, torch.from_numpy(values).float(), [samples, features]
    )
    return indices, t


class ClusterData(torch.utils.data.Dataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
    and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
        save_dir (string, optional): If set, will save the partitioned data to
            the :obj:`save_dir` directory for faster re-use.
    """

    def __init__(self, data:     torch_geometric.data.data.Data, num_parts: int, recursive: bool=False, save_dir: None=None) -> None:
        assert data.edge_index is not None

        self.num_parts = num_parts
        self.recursive = recursive
        self.save_dir = save_dir

        self.process(data)

    def process(self, data:     torch_geometric.data.data.Data) -> None:
        recursive = "_recursive" if self.recursive else ""
        filename = f"part_data_{self.num_parts}{recursive}.pt"

        path = osp.join(self.save_dir or "", filename)
        if self.save_dir is not None and osp.exists(path):
            data, partptr, perm = torch.load(path)
        else:
            data = copy.copy(data)
            num_nodes = data.num_nodes

            (row, col), edge_attr = data.edge_index, data.edge_attr
            adj = SparseTensor(row=row, col=col, value=edge_attr)
            adj, partptr, perm = adj.partition(self.num_parts, self.recursive)

            for key, item in data:
                if item.size(0) == num_nodes:
                    data[key] = item[perm]

            data.edge_index = None
            data.edge_attr = None
            data.adj = adj

            if self.save_dir is not None:
                torch.save((data, partptr, perm), path)

        self.data = data
        self.perm = perm
        self.partptr = partptr

    def __len__(self) -> int:
        return self.partptr.numel() - 1

    def __getitem__(self, idx):
        start = int(self.partptr[idx])
        length = int(self.partptr[idx + 1]) - start

        data = copy.copy(self.data)
        num_nodes = data.num_nodes

        for key, item in data:
            if item.size(0) == num_nodes:
                data[key] = item.narrow(0, start, length)

        data.adj = data.adj.narrow(1, start, length)

        row, col, value = data.adj.coo()
        data.adj = None
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data}, " f"num_parts={self.num_parts})"


class ClusterLoader(torch.utils.data.DataLoader):
    r"""The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
    for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
    and their between-cluster links from a large-scale graph data object to
    form a mini-batch.

    Args:
        cluster_data (torch_geometric.data.ClusterData): The already
            partioned data object.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
    """

    def __init__(self, cluster_data: ClusterData, batch_size: int=1, shuffle: bool=False, **kwargs) -> None:
        class HelperDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(cluster_data)

            def __getitem__(self, idx):
                start = int(cluster_data.partptr[idx])
                length = int(cluster_data.partptr[idx + 1]) - start

                data = copy.copy(cluster_data.data)
                num_nodes = data.num_nodes
                for key, item in data:
                    if item.size(0) == num_nodes:
                        data[key] = item.narrow(0, start, length)

                return data, idx

        def collate(batch):
            data_list = [data[0] for data in batch]
            parts: List[int] = [data[1] for data in batch]
            partptr = cluster_data.partptr

            adj = cat([data.adj for data in data_list], dim=0)

            adj = adj.t()
            adjs = []
            for part in parts:
                start = partptr[part]
                length = partptr[part + 1] - start
                adjs.append(adj.narrow(0, start, length))
            adj = cat(adjs, dim=0).t()
            row, col, value = adj.coo()

            data = cluster_data.data.__class__()
            data.num_nodes = adj.size(0)
            data.edge_index = torch.stack([row, col], dim=0)
            data.edge_attr = value

            ref = data_list[0]
            keys = list(ref.keys())
            keys.remove("adj")

            for key in keys:
                if ref[key].size(0) != ref.adj.size(0):
                    data[key] = ref[key]
                else:
                    data[key] = torch.cat(
                        [d[key] for d in data_list], dim=ref.__cat_dim__(key, ref[key])
                    )

            return data

        super(ClusterLoader, self).__init__(
            HelperDataset(), batch_size, shuffle, collate_fn=collate, **kwargs
        )


## model
class GAT(torch.nn.Module):  # torch.nn.Module is the base class for all NN modules.
    def __init__(self, n_nodes: int, nFeatures: int, nHiddenUnits: int, nHeads: int, alpha: float, dropout: float) -> None:
        super(GAT, self).__init__()
        # 定义实例属性
        self.n_nodes = n_nodes
        self.nFeatures = nFeatures
        self.nHiddenUnits = nHiddenUnits
        self.nHeads = nHeads
        self.alpha = alpha
        self.dropout = dropout

        self.gat1 = GATConv(
            self.nFeatures,
            out_channels=self.nHiddenUnits,  # 映射到8维
            heads=self.nHeads,
            concat=True,
            negative_slope=self.alpha,
            dropout=self.dropout,
            bias=True,
        )
        self.gat2 = GATConv(
            self.nHiddenUnits * self.nHeads,
            self.n_nodes,  # 最后一层映射到k维度（k=n_class）
            heads=self.nHeads,
            concat=False,
            negative_slope=self.alpha,
            dropout=self.dropout,
            bias=True,
        )

    def forward(self, data:     torch_geometric.data.data.Data) ->     torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)  # 第一层输出经过ELU非线性函数
        x = F.elu(x)
        x = self.gat2(x, edge_index)  # 第二层输出经过softmax变成[0, 1]后直接用于分类
        # return F.log_softmax(x, dim=1)
        return x


## sklearn classifier
class GATclassifier(BaseEstimator):
    """A pytorch regressor"""

    def __init__(
        self,
        n_nodes: int=2,
        nFeatures: Optional[int]=None,
        nHiddenUnits: int=8,
        nHeads: int=8,
        alpha: float=0.2,
        dropout: float=0.4,
        clip: None=None,
        rs: int=random.randint(1, 1000000),
        LR: float=0.001,
        WeightDecay: float=5e-4,
        BatchSize: int=256,
        NumParts: int=200,
        nEpochs: int=100,
        fastmode: bool=True,
        verbose: int=0,
        device: str="cpu",
    ) -> None:
        """
        Called when initializing the regressor
        """
        self._history = None
        self._model = None

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.copy().items():
            setattr(self, arg, val)

    def _build_model(self) -> None:

        self._model = GAT(
            self.n_nodes,
            self.nFeatures,
            self.nHiddenUnits,
            self.nHeads,
            self.alpha,
            self.dropout,
        )

    def _train_model(self, X: ndarray, y: Categorical, adj: csr_matrix) -> None:
        # X, y, adj = get_X_y_from_ann(adata_train_final, return_adj=True, n_pc=2, n_neigh=10)

        node_features = torch.from_numpy(X).float()
        labels = torch.LongTensor(y)
        edge_index, _ = scipysparse2torchsparse(adj)

        d = Data(x=node_features, edge_index=edge_index, y=labels)
        cd = ClusterData(d, num_parts=self.NumParts)

        cl = ClusterLoader(cd, batch_size=self.BatchSize, shuffle=True)

        optimizer = torch.optim.Adagrad(
            self._model.parameters(), lr=self.LR, weight_decay=self.WeightDecay
        )

        # Random Seed
        random.seed(self.rs)
        np.random.seed(self.rs)
        torch.manual_seed(self.rs)

        t_total = time.time()
        loss_values = []
        bad_counter = 0
        best = self.nEpochs + 1
        best_epoch = 0

        for epoch in range(self.nEpochs):

            t = time.time()
            epoch_loss = []
            epoch_acc = []
            epoch_acc_val = []
            epoch_loss_val = []

            self._model.train()  # It sets the mode to train

            for batch in cl:  # cl: clusterLoader
                batch = batch.to(self.device)  # move the data to CPU/GPU
                optimizer.zero_grad()  # weight init
                x_output = self._model(batch)  # ncell*2; log_softmax
                output = F.log_softmax(x_output, dim=1)

                loss = F.nll_loss(
                    input=output, target=batch.y
                )  # compute negative log likelihood loss
                # input: ncell*nclass;
                # target: ncell*1, 0 =< value <= nclass-1
                loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clip)
                optimizer.step()
                epoch_loss.append(loss.item())
                # epoch_acc.append(accuracy(output, batch.y).item())

            if not self.fastmode:
                d_val = Data(x=features_val, edge_index=edge_index_val, y=labels_val)
                d_val = d_val.to(self.device)
                self._model.eval()
                x_output = self._model(d_val)
                output = F.log_softmax(x_output, dim=1)

                loss_val = F.nll_loss(output, d_val.y)
                # acc_val = accuracy(output,d_val.y).item() # tensor.item() returns the value of this tensor as a standard Python number.
                if (self.verbose > 0) & ((epoch + 1) % 50 == 0):
                    print(
                        "Epoch {}\t<loss>={:.4f}\tloss_val={:.4f}\tin {:.2f}-s".format(
                            epoch + 1,
                            np.mean(epoch_loss),
                            loss_val.item(),
                            time.time() - t,
                        )
                    )
                loss_values.append(loss_val.item())
            else:
                if (self.verbose > 0) & ((epoch + 1) % 50 == 0):
                    print(
                        "Epoch {}\t<loss>={:.4f}\tin {:.2f}-s".format(
                            epoch + 1, np.mean(epoch_loss), time.time() - t
                        )
                    )
                loss_values.append(np.mean(epoch_loss))

    def fit(self, X: ndarray, y: Categorical, adj: csr_matrix) -> "GATclassifier":
        """
        Trains the pytorch regressor.
        """

        self._build_model()
        self._train_model(X, y, adj)

        return self

    def predict(self, X: ndarray, y: Categorical, adj: csr_matrix) ->     torch.Tensor:
        """
        Makes a prediction using the trained pytorch model
        """

        # X, y, adj = get_X_y_from_ann(adata_test, return_adj=True, n_pc=2, n_neigh=10)

        node_features = torch.from_numpy(X).float()
        labels = torch.LongTensor(y)
        edge_index, _ = scipysparse2torchsparse(adj)

        d_test = Data(x=node_features, edge_index=edge_index, y=labels)

        self._model.eval()  # define the evaluation mode
        d_test = d_test.to(self.device)
        x_output = self._model(d_test)
        output = F.log_softmax(x_output, dim=1)
        preds = output.max(1)[1].type_as(labels)

        return preds

    def predict_proba(self, X: ndarray, y: Categorical, adj: csr_matrix) -> ndarray:

        # X, y, adj = get_X_y_from_ann(adata_test, return_adj=True, n_pc=2, n_neigh=10)

        node_features = torch.from_numpy(X).float()
        labels = torch.LongTensor(y)
        edge_index, _ = scipysparse2torchsparse(adj)

        d_test = Data(x=node_features, edge_index=edge_index, y=labels)

        self._model.eval()  # define the evaluation mode
        d_test = d_test.to(self.device)
        x_output = self._model(d_test)
        output = F.log_softmax(x_output, dim=1)

        probs = torch.exp(output)  # return softmax (output is logsoftmax)
        y_prob = (
            probs.detach().cpu().numpy()
        )  # detach() here prune away the gradients bond with the probs tensor

        return y_prob

    def score(self, X, y, sample_weight=None):
        """
        Scores the data using the trained pytorch model. Under current implementation
        returns negative mae.
        """
        y_pred = self.predict(X, y)
        return F.nll_loss(y_pred, y)
