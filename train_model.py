# train_model.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
data = pd.read_csv("your_dataset.csv")  # Ensure it has "SMILES" and "pIC50" columns
smiles_list = data["SMILES"]
targets = data["pIC50"]

# Function to convert SMILES to Morgan fingerprint
def smiles_to_fp(smiles, radius=2, nBits=2048):
    features = []
    valid_targets = []
    for smi, target in zip(smiles, targets):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            features.append(np.array(fp))
            valid_targets.append(target)
    return np.array(features), np.array(valid_targets)

X, y = smiles_to_fp(smiles_list)

# Train a simple model using only built-in classes
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model (✅ this will work across environments)
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
