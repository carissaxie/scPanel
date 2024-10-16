# scPanel

scPanel selects **a sparse gene panel** from responsive cell population(s) for **patient-level classification** in single cell RNA sequencing (**scRNA-seq**) data. 

scPanel can give you:

- Cell populations that are responding to the perturbation (e.g. disease, drugs)

- A minimal number of genes that can discriminate two differerent status in the selected cell population(s)

- A classifier that can predict patients in two differerent status

Specifically, patients are splitted into training and testing set. In the training set, cell populations responsive to perturbations are scored by quantifying how well each cell populations are separeted between two conditions. With the selected population, Support Vector Machine Recursive Feature Elimination (*SVM*-*RFE*) is applied to identify a minimal number of genes with high predictive power. The number of genes in the panel is automatically decided in a data driven way to aviod bias from manual inspection. Using the selected cell population(s) and corresponding gene panel(s), scPanel constructs a patient-level classifier with the training data and evalute its performance in the testing data to validate the power of identified genes. All the data splitting involved in scPanel is done at patient-level so that the importance of selected cell population, genes and the performance of corresponding classifiers are genearalizable to all patients.

## Why scPanel is better:

- Reduce the cost of sequencing with a small number of genes needed for assay

- The number of genes in the panel is automatically decided in a data driven way

- Genearalizable patient-level classification

- Compatible with Scanpy/Anndata framework

# Documentation
Update (23/08/2024)
Documentation has been initialized and will be updated with more details soon. Check out the [scPanel documentation](https://scpanel.readthedocs.io/en/latest/autoapi/scpanel/index.html).

# Usage

scPanel is mainly composed of three steps:

1. idenitfy responsive cell population

2. identify a sparse gene panel

3. patient-level classification

## Input scRNA-seq data

1. Quality control and preprocess data using standard workflow.

2. Annotate cell populations.

3. Input AnnData Object to scPanel.

## Functions

- `preprocess`(adata, ct_col, y_col, pt_col, class_map)
  
  - standardize metadata

- `split_train_test`(adata, out_dir, min_cells=20, min_samples=3, test_pt_size=0.2, random_state=3467, verbose=0)
  
  - split patients into 1) training set for cell type selection, gene panel identification and classfiers training, 2) testing set to evalute the performance of classifiers and validate the predictive power of the gene panel.

- `cell_type_score`(adata_train_dict, out_dir, ncpus, n_iterations, sample_n_cell, n_iterations=100, verbose=False)
  
  - calculate cell type responsive score (AUC) for each cell population annotated.

- `plot_cell_type_score`(AUC, AUC_all)
  
  - visualize cell type responsive score (AUC)

- `select_celltype`(adata_train_dict, out_dir, celltype_selected)
  
  - prepare anndata for gene panel selection

- `split_n_folds`(adata_train, nfold, out_dir=None, random_state=2349)
  
  - split data into multiple folds for gene selection

- `gene_score`(adata_train, train_index_list, val_index_list, sample_weight_list, metric, out_dir, ncpus, step=0.03, verbose = False)
  
  - scoring genes by their predictive power

- `decide_k`(adata_train, n_genes_plot=100)
  
  - automatically decide the number of genes selected for patient classification

- `plot_gene_score`(adata_train, n_genes_plot = 200, width=5, height=4, k=None)
  
  - visualize the gene score

- `select_gene`(adata_train, top_n_feat, out_dir=None, step=0.03, n_genes_plot=100, verbose=0)
  
  - select the top K (returned by `decide_k`) genes from the training set

- `transform_adata`(adata_train, adata_test_dict, selected_gene)
  
  - subset the training and testing set with the selected cell population and genes

- `models_train`(adata_train_final, search_grid, out_dir=None, param_grid=None)
  
  - train classifiers with LR, KNN, RF, SVM, GAT

- `models_predict`(clfs, adata_test_final, out_dir=None)
  
  - predict the probablities of cells in the testing set

- `pt_pred`(adata_test_final, cell_pred_col = 'median_pred_score', num_bootstrap=None)
  
  - predict the patient label in the testing set

- `plot_roc_curve`(adata_test_final, sample_id, cell_pred_col, ncols = 4, hspace = 0.25, wspace = None, ax = None, scatter_kws = None, legend_kws = None)
  
  - visulize the aggregation of cell-level probabilities to patient-level label using area under the curve

- `plot_violin`(adata, cell_pred_col = 'median_pred_score', dot_size = 2, ax=None, palette=None, xticklabels_color=False, text_kws={})
  
  - visualize the patient-level prediction

# Citation

If you use `scPanel` in your work, please cite the `scPanel` publication:

Xie, Yi, et al. "scPanel: A tool for automatic identification of sparse gene panels for generalizable patient classification using scRNA-seq datasets." *bioRxiv* (2024): 2024-04.


