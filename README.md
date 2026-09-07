# Blood-Brain Barrier Permeability Prediction

Machine learning pipeline for predicting whether small molecules cross the blood-brain barrier (BBB) from molecular structure.

This project uses molecular fingerprints and physicochemical descriptors to compare multiple classification approaches. Molecules are separated using Bemis-Murcko scaffolds rather than a conventional random split to provide a more challenging evaluation of generalization to structurally distinct compounds.

## Results

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| Logistic Regression | ECFP4 | 0.835 | 0.940 |
| Random Forest | ECFP4 | 0.889 | 0.963 |
| LightGBM | ECFP4 + RDKit descriptors | **0.901** | **0.971** |

Metrics are reported on a held-out 20% scaffold-based test set.

## Dataset

The project uses the MoleculeNet BBBP (Blood-Brain Barrier Penetration) benchmark, containing molecular structures represented as SMILES strings with binary BBB permeability labels.

After molecular parsing with RDKit, **2,039 molecules** were retained for modeling.

The dataset is class-imbalanced, with BBB-penetrant molecules representing the majority class.

## Molecular Representation

Two representations are used:

- **ECFP4 fingerprints** — 1,024-bit Morgan fingerprints encoding local molecular substructures
- **RDKit descriptors** — 200+ physicochemical and structural molecular properties

Logistic regression and random forest models use the ECFP4 representation as baselines.

The final LightGBM model combines the fingerprints with RDKit descriptors.

## Scaffold-Based Evaluation

A random molecular split can place structurally similar compounds in both the training and test sets.

To reduce this structural overlap, molecules are grouped using their **Bemis-Murcko scaffolds**, and complete scaffold families are assigned to either training or testing.

Final split:

- **1,631 training molecules**
- **408 test molecules**
- **No scaffold families shared between training and test sets**

This provides a more challenging estimate of performance on structurally distinct molecules.

## Modeling

Three models are compared:

1. **Logistic Regression** — linear ECFP4 baseline
2. **Random Forest** — nonlinear ECFP4 baseline
3. **LightGBM** — gradient-boosted trees using ECFP4 fingerprints and RDKit molecular descriptors

The final LightGBM model achieved a held-out **ROC-AUC of 0.901** and **PR-AUC of 0.971**.

The complete modeling workflow is available in [`bbbp_modeling_notebook.ipynb`](bbbp_modeling_notebook.ipynb).

## Running the Predictor

The trained LightGBM model and descriptor scaler are included in the repository:

```text
bbb_lightgbm_model.pkl
bbb_descriptor_scaler.pkl
```

Create and activate a virtual environment:

```bash
python3 -m venv mol-ml-env
source mol-ml-env/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the default example:

```bash
python predict.py
```

Or provide a molecule as a SMILES string:

```bash
python predict.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

The script reports the model's predicted BBB penetration probability and selected molecular properties such as TPSA, MolLogP, and hydrogen-bond donors.

## Repository Structure

```text
bbb-prediction/
├── bbbp_modeling_notebook.ipynb  # Final modeling workflow
├── bbb_lightgbm_model.pkl         # Trained LightGBM classifier
├── bbb_descriptor_scaler.pkl      # Fitted descriptor scaler
├── predict.py                     # Command-line inference script
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

Local development files, including the virtual environment and exploratory notebook, are excluded from version control.

## Tech Stack

**Python · RDKit · scikit-learn · LightGBM · NumPy · pandas · Matplotlib**

## Limitations

- BBBP is a relatively small and class-imbalanced molecular dataset.
- Reported performance is based on one fixed scaffold-based train/test split rather than repeated scaffold cross-validation.
- Molecular fingerprints are hashed representations and should not be interpreted as causal chemical explanations.
- Predicted probabilities are computational model estimates, not experimental measurements of BBB permeability.

## Reproducibility

The final notebook contains the full workflow from molecular parsing and featurization through scaffold splitting, model training, evaluation, and interpretation.

The serialized LightGBM model and fitted descriptor scaler are provided for standalone inference through `predict.py`.