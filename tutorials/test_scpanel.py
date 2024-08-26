import sys
from scpanel.utils_func import *
from scpanel.split_patient import *
from scpanel.select_cell import *
from scpanel.select_gene import *
from scpanel.train import *
from scpanel.settings import *
import anndata

dataset = 'wilk2020covid'
out_dir = './test_result/'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

fastmode = True

adata = anndata.read_h5ad('./wilk2020covid_processed_rna_assay_2.h5ad')
adata.obs.columns
adata

## Check what are condition labels in the data and encode it
print(adata.obs[['disease_status_standard']].drop_duplicates())
class_map = {'healthy': 0, 'COVID-19': 1}

## standardize data
adata = preprocess(adata,
                   ct_col='cell.type.fine',
                   y_col='disease_status_standard',
                   pt_col='sample',
                   class_map=class_map)

## split data
adata_train_dict, adata_test_dict = split_train_test(adata, min_cells=20,
                                                     test_pt_size=0.2,
                                                     out_dir=out_dir,
                                                     random_state=3467)

# Cell type selection
## 1. calculate responsiveness score
AUC, AUC_all = cell_type_score(adata_train_dict,
                               out_dir=out_dir,
                               ncpus=16,
                               n_iterations=30,
                               sample_n_cell=20)

## 2. plot responsiveness score
axes = plot_cell_type_score(AUC, AUC_all)
axes.set_xlim([0.4, 1])
plt.savefig(f'{out_dir}/cell_type_score.pdf', bbox_inches="tight")

## 3. select the most responsive cell type for downstream
top_ct = AUC['celltype'].iloc[-1]
adata_train = select_celltype(adata_train_dict,
                              celltype_selected=top_ct,
                              out_dir=out_dir)

# Gene selection
## 1. split training data
train_index_list, val_index_list, sample_weight_list = split_n_folds(adata_train,
                                                                     nfold=5,
                                                                     out_dir=out_dir,
                                                                     random_state=2349)

## 2. score genes
adata_train, rfecv = gene_score(adata_train,
                                train_index_list,
                                val_index_list,
                                sample_weight_list=sample_weight_list,
                                step=0.03,
                                out_dir=out_dir,
                                ncpus=16,
                                verbose=False)

## 3. plot gene scores
plot_gene_score(adata_train, n_genes_plot=200)
plt.savefig(f'{out_dir}/gene_score.pdf', bbox_inches="tight")

## 4. find the optimal number of informative genes
k = decide_k(adata_train, n_genes_plot=100)
plot_gene_score(adata_train, n_genes_plot=100, k=k)
plt.savefig(f'{out_dir}/decide_k.pdf', bbox_inches="tight")

## 5. return the list of informative genes
adata_train = select_gene(adata_train,
                          top_n_feat=k,
                          step=0.03,
                          out_dir=out_dir)

## 6. view the list of informative genes
sig_svm = adata_train.uns['svm_rfe_genes']
print(sig_svm)

# Classification
## 1. Subset training and testing set with selected cell type and genes
adata_train_final, adata_test_final = transform_adata(adata_train,
                                                      adata_test_dict,
                                                      selected_gene=sig_svm)

## 2. models training
# overwrite default parameters for models
param_grid = {'LR': {'max_iter': 600}}

clfs = models_train(adata_train_final,
                    search_grid=False,
                    out_dir=out_dir, param_grid=param_grid)

## 3. models testing
### cell-level prediction
adata_test_final, y_pred_list, y_pred_score_list = models_predict(clfs, adata_test_final, out_dir=out_dir)

### sample-level prediction and evaluation
all_pred_scores = ['LR_pred_score', 'RF_pred_score', 'SVM_pred_score',
                   'KNN_pred_score', 'GAT_pred_score', 'median_pred_score']

for i in tqdm(range(len(all_pred_scores))):
    adata_test_final = pt_pred(adata_test_final,
                               cell_pred_col=all_pred_scores[i],
                               num_bootstrap=1000)

### visualize patient-level AUC scores
sample_id_list = adata_test_final.obs[['patient_id', 'median_pred_score_sample_auc']].drop_duplicates().sort_values(
    by='median_pred_score_sample_auc')['patient_id']
plot_roc_curve(adata_test_final,
               sample_id=sample_id_list,
               cell_pred_col='median_pred_score',
               hspace=0.4, ncols=4,
               scatter_kws={'s': 10}, legend_kws={'prop': {'size': 11}})
plt.savefig(f'{out_dir}/ROC_curve_sample.pdf', bbox_inches="tight")

### visualize cell-level prediction probabilities and patient-level AUC scores
fig, axes = plt.subplots(6, figsize=(13, 30))
for ax, pred in zip(axes, all_pred_scores):
    print('Plotting for ', pred)
    plot_violin(adata_test_final,
                cell_pred_col=pred,
                ax=ax,
                palette={'healthy': 'C0', 'COVID-19': 'C1'})

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=1)

plt.savefig(f"{out_dir}/Violin_pt_pred_prob.pdf", bbox_inches="tight")
plt.show()

### calculate patient-level precision, recall, f1score and accuracy and visualize
for i in range(len(all_pred_scores)):
    adata_test_final = pt_score(adata_test_final, cell_pred_col=all_pred_scores[i])

plt.figure(figsize=(30, 10))
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2")
# Loop to plot
for metric_idx in range(0, len(adata_test_final.uns['sample_score'].columns)):
    make_single_spider(adata_test_final,
                       metric_idx=metric_idx,
                       color='grey',
                       nrow=1, ncol=6)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=1.5,
                    hspace=1)

plt.savefig(f"{out_dir}/SpiderPlot_pt_score.pdf", bbox_inches="tight")
plt.show()
