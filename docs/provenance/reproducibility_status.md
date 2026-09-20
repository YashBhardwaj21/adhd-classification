# Reproducibility and Provenance Status

## Experiment 1

Canonical notebook: `notebooks/exp01/01_fc_generation_and_validation.ipynb`

Dynamic temporal validation, static-vs-dynamic validation, and the dynamic manifest are staged.

The source audit identified an implementation detail concerning the FC diagonal: the executed FC generation retains diagonal self-correlation values.

## Experiment 2

Canonical notebooks:

- `notebooks/exp02/02_graph_construction_and_validation.ipynb`
- `notebooks/exp02/03_graph_metrics_and_null_models.ipynb`

Graph construction code, null-model implementations, configuration, and lightweight result tables are staged.

The large node-level parquet artifact is excluded.

## Experiment 3

Canonical notebook: `notebooks/exp03/04_dynamic_graph_feature_extraction.ipynb`

Feature and statistical outputs were traced during the audit, but not every paper statistic was independently mapped to a standalone public artifact.

## Experiment 4

Canonical notebook: `notebooks/exp04/w2c_athena_2.ipynb`

Principal harmonization, prediction, topology-preservation, and site-prediction outputs were traced during the audit.

## Experiment 5

Canonical notebook: `notebooks/exp05/dynamic_transformer.ipynb`

Dynamic-state discovery outputs were traced at experiment/result level.

Large parquet intermediates are excluded.

## Experiment 6

Canonical notebook: `notebooks/exp06/exp06_semi_supervised_pseudolabeling.ipynb`

Latest verified Procedure I uses SimpleImputer, StandardScaler, class-weighted Logistic Regression, and SelfTrainingClassifier.

Parameters: threshold 0.75, k_best 10, max_iter 10.

Stored execution records 552 newly accepted pseudo-labels.

Procedure II uses Random Forest, Gradient Boosting, Logistic Regression, and calibrated SVM in a weighted ensemble.

The stored execution records 484 selected subjects.

## Experiment 7

Classical GCN and QGCNN source and primary result artifacts are staged.

This experiment uses its own downstream cohort and does not consume the Experiment 6 pseudo-label sets.

## Experiment 8

Staged baselines:

- Lightweight 3D CNN
- NeuroSTORM
- Temporal graph learning

Large NeuroSTORM checkpoints are excluded.

`updated_transformer.ipynb` is not treated as one of the paper-reported Experiment 8 baselines.

## Experiment 9

Canonical notebook: `notebooks/exp09/11_population_graph_learning.ipynb`

The executed LOSO artifacts and configuration files are staged.

The executed configuration differs from the paper description in graph construction.

Executed configuration: top-10% graph selection, connectivity features, weighted edges, and self-loops.

Paper description: MST + 20% graph construction and inverse-distance mapping.

The discrepancy remains unresolved and is preserved explicitly in the repository metadata.
