import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR


def load_dataset(path: str) -> pd.DataFrame:
    """Load merged PFZ dataset."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded dataset with {len(df)} records from {path}")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Build feature matrix and targets."""
    feature_cols = [
        "lat",
        "lon",
        "depth_m",
        "sst_0_10m_mean",
        "sst_0_10m_anomaly",
        "viirs_sst",
        "viirs_chl_a",
    ]

    X = df[feature_cols].copy()
    y_reg = df["cpue_kg_per_h"].astype(float)
    y_cls = df["pfz_binary"].astype(int)

    # Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)

    # Track missing data
    missing_count = X.isnull().sum().sum()
    print(f"Missing values in feature matrix: {missing_count}")

    # Impute with median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols)

    return X, y_reg, y_cls


def time_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Split data by time (not random)."""
    sorted_df = df.sort_values("date").reset_index(drop=True)
    cutoff = int(len(sorted_df) * (1 - test_fraction))
    train_df = sorted_df.iloc[:cutoff]
    test_df = sorted_df.iloc[cutoff:]
    print(f"\nTime-based split:")
    print(f"  Training: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Testing:  {test_df['date'].min()} to {test_df['date'].max()}")
    return train_df, test_df


def evaluate_regression(y_true, y_pred, prefix=""):
    """Evaluate regression model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"\n{prefix} Regression Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")
    return {"rmse": float(rmse), "r2": float(r2), "mae": float(mae)}


def evaluate_classification(y_true, y_pred, prefix=""):
    """Evaluate classification model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{prefix} Classification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"    FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PFZ forecasting models")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to merged dataset CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(MODELS_DIR), help="Output directory"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PFZ Model Training")
    print("=" * 60)

    # Load data
    df = load_dataset(args.dataset)

    # Time-based split
    train_df, test_df = time_split(df, test_fraction=0.2)

    # Build features
    print("\nBuilding feature matrices...")
    X_train, y_reg_train, y_cls_train = build_feature_matrix(train_df)
    X_test, y_reg_test, y_cls_test = build_feature_matrix(test_df)

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"PFZ positive class in train: {(y_cls_train == 1).sum()}")
    print(f"PFZ positive class in test: {(y_cls_test == 1).sum()}")

    # ========== REGRESSION MODEL ==========
    print("\n" + "=" * 60)
    print("Training Regression Model (CPUE Prediction)")
    print("=" * 60)

    reg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    verbose=1,
                ),
            ),
        ]
    )

    print("Fitting regression model...")
    reg_pipe.fit(X_train, y_reg_train)
    reg_pred_train = reg_pipe.predict(X_train)
    reg_pred_test = reg_pipe.predict(X_test)

    reg_metrics_train = evaluate_regression(y_reg_train, reg_pred_train, "Training")
    reg_metrics_test = evaluate_regression(y_reg_test, reg_pred_test, "Testing")

    # Feature importance
    reg_model = reg_pipe.named_steps["model"]
    reg_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": reg_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nRegression Model Feature Importance:")
    print(reg_importance.to_string(index=False))

    # ========== CLASSIFICATION MODEL ==========
    print("\n" + "=" * 60)
    print("Training Classification Model (PFZ Detection)")
    print("=" * 60)

    cls_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    verbose=1,
                ),
            ),
        ]
    )

    print("Fitting classification model...")
    cls_pipe.fit(X_train, y_cls_train)
    cls_pred_train = cls_pipe.predict(X_train)
    cls_pred_test = cls_pipe.predict(X_test)

    cls_metrics_train = evaluate_classification(
        y_cls_train, cls_pred_train, "Training"
    )
    cls_metrics_test = evaluate_classification(y_cls_test, cls_pred_test, "Testing")

    # Feature importance
    cls_model = cls_pipe.named_steps["model"]
    cls_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": cls_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nClassification Model Feature Importance:")
    print(cls_importance.to_string(index=False))

    # ========== SAVE MODELS ==========
    print("\n" + "=" * 60)
    print("Saving Models")
    print("=" * 60)

    joblib.dump(reg_pipe, output_dir / "pfz_regression_model.joblib")
    print(f"Saved regression model to {output_dir / 'pfz_regression_model.joblib'}")

    joblib.dump(cls_pipe, output_dir / "pfz_classification_model.joblib")
    print(f"Saved classification model to {output_dir / 'pfz_classification_model.joblib'}")

    # ========== SAVE RESULTS ==========
    result = {
        "training_period": {
            "start": train_df["date"].min().strftime("%Y-%m-%d"),
            "end": train_df["date"].max().strftime("%Y-%m-%d"),
        },
        "test_period": {
            "start": test_df["date"].min().strftime("%Y-%m-%d"),
            "end": test_df["date"].max().strftime("%Y-%m-%d"),
        },
        "training_set_size": len(train_df),
        "test_set_size": len(test_df),
        "regression_metrics_train": reg_metrics_train,
        "regression_metrics_test": reg_metrics_test,
        "classification_metrics_train": cls_metrics_train,
        "classification_metrics_test": cls_metrics_test,
        "regression_feature_importance": reg_importance[["feature", "importance"]]
        .to_dict(orient="records"),
        "classification_feature_importance": cls_importance[["feature", "importance"]]
        .to_dict(orient="records"),
    }

    metrics_file = output_dir / "pfz_model_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
