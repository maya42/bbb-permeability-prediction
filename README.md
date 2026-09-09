# Blood-Brain Barrier Permeability Prediction

An independent molecular machine learning project using public MoleculeNet
BBBP data to predict blood-brain barrier penetration from molecular structure.

## Research question

Do RDKit physicochemical descriptors improve predictions beyond ECFP4
fingerprints under fixed LightGBM settings?

The project benchmarks several classifiers, then compares three molecular
representations using five-fold scaffold-grouped cross-validation.

## Dataset and molecular features

The MoleculeNet BBBP benchmark contains SMILES strings with binary
BBB penetration labels. After RDKit parsing, **2,039 molecules** were
retained. BBB-penetrant molecules represent the majority class.

Two feature representations are used:

- **ECFP4 fingerprints:** 1,024-bit Morgan fingerprints with radius 2.
- **RDKit descriptors:** 210 physicochemical and structural descriptors
  in the recorded run.

Dataset: [MoleculeNet BBBP CSV](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv)

## Evaluation procedure

Molecules are grouped by Bemis–Murcko scaffold so that molecules sharing
a scaffold remain in the same partition.

The original split contains:

- **1,631 training molecules**
- **408 test molecules**
- **Zero shared scaffolds between training and test sets**

The feature comparison uses five-fold `StratifiedGroupKFold` within the
original training set, with `shuffle=True` and `random_state=42`.
Scaffold groups remain intact; class balance is approximate.
The original test molecules are excluded from cross-validation.

All three LightGBM feature configurations use identical folds and settings:

- 300 boosting trees
- Learning rate: 0.03
- Maximum depth: 6
- Random seed: 42
- One training thread

Descriptor scaling is fitted separately on each fold's training portion.
The negative-to-positive class-weight ratio is also calculated from that
portion only.

## Cross-validation results

Values are **mean ± sample standard deviation** across five validation
folds, rounded to three decimal places. AP denotes average precision.

| LightGBM features | ROC-AUC | AP |
|---|---:|---:|
| ECFP4 fingerprints only | 0.864 ± 0.048 | 0.945 ± 0.021 |
| RDKit descriptors only | 0.893 ± 0.018 | 0.955 ± 0.012 |
| Fingerprints + descriptors | **0.899 ± 0.024** | **0.957 ± 0.013** |

Adding descriptors to fingerprints improved ROC-AUC in four of five
folds, with an average increase of approximately **0.035**.

Descriptors alone performed nearly as well as the combined representation.
Adding fingerprints to descriptors increased average ROC-AUC by
approximately **0.006**.

Under these fixed settings, descriptors contributed useful predictive
information beyond fingerprints. Combined features had the highest
average scores but did not outperform both alternatives in every fold.

## Original single-split results

The initial workflow compared logistic regression, random forest, and
combined-feature LightGBM on the original scaffold test set.

| Model | Features | ROC-AUC | AP |
|---|---|---:|---:|
| Logistic regression | ECFP4 | 0.835 | 0.940 |
| Random forest | ECFP4 | 0.889 | 0.963 |
| LightGBM | ECFP4 + RDKit descriptors | 0.901 | 0.971 |

These results represent a different evaluation procedure from the
cross-validation comparison above. The original test set was previously
inspected and is not a newly untouched evaluation set.

The initial comparison changes both classifier and feature representation;
the controlled LightGBM comparison isolates the feature choice.

## Running the notebook

Create and activate a virtual environment on macOS/Linux:

```bash
python3 -m venv mol-ml-env
source mol-ml-env/bin/activate
python -m pip install -r requirements.txt
```

Open [`bbbp_modeling_notebook.ipynb`](bbbp_modeling_notebook.ipynb)
in a notebook editor, select the environment, and run the cells in order.

The notebook includes:

1. Data loading and molecular featurization
2. The original scaffold split and classifier benchmarks
3. Fifteen cross-validation fits: three feature configurations across five folds
4. Per-fold metrics, summary statistics, and paired ROC-AUC differences
5. Feature importance and training-set Morgan-bit associations
6. An example molecule-screening utility

Section 5b prints progress during cross-validation and produces
`cv_results`, `cv_summary`, and `paired_auc`.

Internet access is required to download the dataset.

## Running the predictor

The standalone predictor uses the repository's saved LightGBM model and
descriptor scaler.

Run the default example:

```bash
python predict.py
```

Or supply a SMILES string:

```bash
python predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

The script reports a model-estimated BBB penetration probability and
selected molecular properties, including TPSA, MolLogP, and
hydrogen-bond donors.

## Repository files

| File | Purpose |
|---|---|
| `bbbp_modeling_notebook.ipynb` | Modeling, evaluation, and interpretation |
| `bbb_lightgbm_model.pkl` | Saved LightGBM classifier |
| `bbb_descriptor_scaler.pkl` | Saved descriptor scaler |
| `predict.py` | Command-line prediction utility |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview and instructions |

## Limitations

- BBBP is a small dataset imbalanced toward BBB-penetrant molecules.
- Cross-validation uses one five-fold scaffold partition. Additional
  partitions and external validation would help assess robustness.
- Fold standard deviations describe variability, not confidence
  intervals. Observed differences do not establish statistical significance.
- Class prevalence varies across folds and should be considered when
  interpreting AP.
- Scaffold separation prevents shared scaffold groups but does not
  establish generalization across all chemical space.
- Feature importance and hashed Morgan-bit associations are not evidence
  of causal molecular effects.
- Prediction scores are computational estimates, not experimentally
  validated measurements of BBB permeability.

## Reproducibility

Split and model random seeds are fixed at 42. Cross-validation fits
preprocessing separately within each training fold.

Reproducing the recorded results also requires consistent data and package
versions. RDKit versions can affect the available descriptor set.
The standalone predictor must use the same feature definitions and
descriptor ordering as its saved model and scaler.

## Tech stack

Python · RDKit · scikit-learn · LightGBM · NumPy · pandas · Matplotlib