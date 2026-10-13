"""Leakage-safe feature engineering example using scikit-learn."""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ["age", "annual_income", "visits_per_month"]
CATEGORICAL_FEATURES = ["membership", "region"]


def make_synthetic_dataset(n_rows: int = 500, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Build deterministic synthetic tabular data with missing values."""
    if n_rows < 30:
        raise ValueError("n_rows must be at least 30 for a meaningful split.")
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 76, n_rows).astype(float)
    income = np.clip(rng.normal(55_000, 17_000, n_rows), 12_000, None)
    visits = rng.poisson(5, n_rows).astype(float)
    membership = rng.choice(["basic", "plus", "premium"], n_rows, p=[0.5, 0.32, 0.18]).astype(object)
    region = rng.choice(["north", "south", "east", "west"], n_rows).astype(object)

    score = (
        0.025 * (age - 40)
        + 0.000025 * (income - 50_000)
        + 0.22 * (visits - 5)
        + np.where(membership == "premium", 0.65, np.where(membership == "plus", 0.25, 0.0))
        + rng.normal(0, 0.85, n_rows)
    )
    target = pd.Series((score > np.median(score)).astype(int), name="target")

    # Inject missing values in features only, after target generation.
    age[rng.choice(n_rows, max(1, n_rows // 25), replace=False)] = np.nan
    income[rng.choice(n_rows, max(1, n_rows // 20), replace=False)] = np.nan
    membership[rng.choice(n_rows, max(1, n_rows // 30), replace=False)] = None
    region[rng.choice(n_rows, max(1, n_rows // 35), replace=False)] = None

    features = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "visits_per_month": visits,
        "membership": membership,
        "region": region,
    })
    return features, target


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create column-specific transformations without fitting on any data."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(scale_numeric: bool = True) -> Pipeline:
    """Return a full preprocessing + baseline classifier pipeline."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def run_demo(output_dir: str | Path = "outputs", seed: int = 42) -> dict[str, float | int]:
    """Split first, fit only on train data, evaluate and save previews."""
    X, y = make_synthetic_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_test = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transformed_test, columns=feature_names).head(20).to_csv(
        output_path / "transformed_test_preview.csv", index=False
    )
    pd.DataFrame({"feature_name": feature_names}).to_csv(
        output_path / "transformed_feature_names.csv", index=False
    )

    metrics: dict[str, float | int] = {
        "rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "raw_features": X.shape[1],
        "transformed_features": transformed_test.shape[1],
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    print(f"Rows: {metrics['rows']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Raw input features: {metrics['raw_features']}")
    print(f"Transformed features: {metrics['transformed_features']}")
    print(f"Test accuracy (synthetic demo): {metrics['test_accuracy']:.3f}")
    print(f"Test F1 (synthetic demo): {metrics['test_f1']:.3f}")
    print(f"Outputs saved under: {output_path.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs", help="Directory for output CSVs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    run_demo(output_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
