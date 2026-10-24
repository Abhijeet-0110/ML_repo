"""
train.py
--------
Trains a RandomForestClassifier on synthetic plant care data.
Handles one-hot encoding, feature alignment persistence to avoid inference
mismatches, evaluation metrics, and model artifact saving.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Safe directory resolution relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "plants.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "plant_model.joblib"
COLUMNS_FILE = MODELS_DIR / "model_columns.joblib"
METADATA_FILE = MODELS_DIR / "metadata.joblib"

CATEGORICAL_COLS = ["plant_type", "humidity", "has_drainage"]
NUMERICAL_COLS = [
    "watering_per_week",
    "sunlight_hours",
    "temperature_c",
    "forgetfulness_score",
]
TARGET_COL = "outcome"
TARGET_CLASSES = ["Thrives", "Survives", "Doomed"]


def load_and_preprocess_data():
    """
    Loads dataset, checks for nulls, and applies one-hot encoding.
    Returns:
        X_encoded (pd.DataFrame): Encoded features
        y (pd.Series): Target labels
        feature_columns (list): Order of feature columns after encoding
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_FILE}'. "
            f"Please run 'python src/generate_data.py' first."
        )

    df = pd.read_csv(DATA_FILE)
    print(f"[+] Loaded {len(df):,} records from {DATA_FILE}")

    # Verify no missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[!] Warning: Found {missing_count} missing values. Dropping rows with nulls.")
        df = df.dropna()

    X = df[CATEGORICAL_COLS + NUMERICAL_COLS].copy()
    y = df[TARGET_COL].copy()

    # One-hot encode categorical features (keep all levels for clear feature alignment)
    X_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=False)

    # Convert boolean dummy columns to int (0/1) for compatibility
    dummy_cols = [c for c in X_encoded.columns if c not in NUMERICAL_COLS]
    X_encoded[dummy_cols] = X_encoded[dummy_cols].astype(int)

    feature_columns = list(X_encoded.columns)
    return X_encoded, y, feature_columns


def train_and_evaluate():
    """
    Splits data 80/20 with stratification, fits RandomForestClassifier,
    reports comprehensive metrics, and saves model and column schema.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_columns = load_and_preprocess_data()

    # 80/20 train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[+] Dataset split: {len(X_train):,} train samples, {len(X_test):,} test samples")

    # Train Random Forest Classifier
    print("[*] Training RandomForestClassifier (n_estimators=200, random_state=42)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 55)

    # Sanity check on accuracy target (80% - 90%)
    if 0.80 <= accuracy <= 0.90:
        print("[SUCCESS] Accuracy is within the target realistic window (80-90%).")
    elif accuracy > 0.90:
        print(
            "[NOTICE] Accuracy is slightly above 90%. "
            "Consider adding more noise in src/generate_data.py if strict 80-90% is needed."
        )
    else:
        print(
            "[NOTICE] Accuracy is below 80%. "
            "Consider decreasing noise in src/generate_data.py."
        )

    # Confusion Matrix
    print("\n[+] Confusion Matrix (Rows = Actual, Columns = Predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)
    cm_df = pd.DataFrame(cm, index=TARGET_CLASSES, columns=TARGET_CLASSES)
    print(cm_df.to_string())

    # Classification Report
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, labels=TARGET_CLASSES, digits=4))

    # Feature Importances
    print("[+] Feature Importances:")
    importances = pd.Series(clf.feature_importances_, index=feature_columns).sort_values(
        ascending=False
    )
    for feat, imp in importances.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<30} {imp:6.4f} | {bar}")

    # Save artifacts using joblib
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(feature_columns, COLUMNS_FILE)

    metadata = {
        "accuracy": float(accuracy),
        "target_classes": TARGET_CLASSES,
        "feature_columns": feature_columns,
        "categorical_cols": CATEGORICAL_COLS,
        "numerical_cols": NUMERICAL_COLS,
        "feature_importances": importances.to_dict(),
    }
    joblib.dump(metadata, METADATA_FILE)

    print(f"\n[+] Model saved to:         {MODEL_FILE}")
    print(f"[+] Column order saved to:  {COLUMNS_FILE}")
    print(f"[+] Metadata saved to:      {METADATA_FILE}")


if __name__ == "__main__":
    train_and_evaluate()
