# =====================================================
# TRAIN XGBOOST MODEL - PRODUCTION SAFE VERSION
# =====================================================

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import os

# -----------------------------------------------------
# 1️⃣ DETECT PROJECT ROOT
# -----------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

print("Project Root:", BASE_DIR)

# -----------------------------------------------------
# 2️⃣ LOAD DATA
# -----------------------------------------------------

DATA_PATH = BASE_DIR / "data" / "processed" / "final_dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError("final_dataset.csv not found in data/processed/")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# -----------------------------------------------------
# 3️⃣ SPLIT FEATURES & LABEL
# -----------------------------------------------------

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train Shape:", X_train.shape)

# -----------------------------------------------------
# 4️⃣ TRAIN XGBOOST
# -----------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

print("Training Complete.")

# -----------------------------------------------------
# 5️⃣ EVALUATE
# -----------------------------------------------------

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# 6️⃣ SAVE MODEL (SAFE METHOD)
# -----------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_model.json"

booster = model.get_booster()
booster.save_model(str(MODEL_PATH))

print("Model Saved At:", MODEL_PATH)
print("File Size (KB):", os.path.getsize(MODEL_PATH) / 1024)

# -----------------------------------------------------
# 7️⃣ SAVE TEST DATA FOR SHAP
# -----------------------------------------------------

X_test.to_csv(BASE_DIR / "data" / "processed" / "X_test.csv", index=False)

print("X_test saved for SHAP.")