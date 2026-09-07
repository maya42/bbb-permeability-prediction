import sys
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# Load trained model artifacts
BASE_DIR = Path(__file__).resolve().parent

lgb_model = joblib.load(BASE_DIR / "bbb_lightgbm_model.pkl")
scaler = joblib.load(BASE_DIR / "bbb_descriptor_scaler.pkl")


def screen_smiles(smiles: str):
    """Predict BBB permeability from a SMILES string."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Error: Invalid SMILES string '{smiles}'")
        return

    # 1. Generate 1024-bit ECFP4 / Morgan fingerprint
    fp = np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=1024,
        )
    ).reshape(1, -1)

    # 2. Calculate RDKit descriptors
    desc_values = [
        func(mol)
        for _, func in Descriptors._descList
    ]

    desc_clean = np.nan_to_num(
        np.array(desc_values, dtype=float).reshape(1, -1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Apply the scaler fitted during training
    desc_scaled = scaler.transform(desc_clean)

    # 3. Combine fingerprint + descriptors
    X_input = np.hstack([fp, desc_scaled])

    # Check that the feature vector matches the trained model
    if X_input.shape[1] != lgb_model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: model expects "
            f"{lgb_model.n_features_in_} features, "
            f"but generated {X_input.shape[1]}."
        )

    # 4. Predict probability
    prob = lgb_model.predict_proba(X_input)[0, 1]

    # 5. Report
    print("=" * 55)
    print("        BLOOD-BRAIN BARRIER SCREEN")
    print("=" * 55)
    print(f"SMILES:              {smiles}")
    print(f"Penetration Chance:  {prob:.1%}")
    print(
        f"Prediction:          "
        f"{'PENETRANT' if prob >= 0.5 else 'NON-PENETRANT'}"
    )
    print("-" * 55)

    print("Molecular Properties")
    print(f"TPSA:                {Descriptors.TPSA(mol):.1f} Å²")
    print(f"MolLogP:             {Descriptors.MolLogP(mol):.2f}")
    print(f"H-bond donors:       {Descriptors.NumHDonors(mol)}")
    print("=" * 55)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        smiles = " ".join(sys.argv[1:])
        screen_smiles(smiles)
    else:
        screen_smiles(
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        )