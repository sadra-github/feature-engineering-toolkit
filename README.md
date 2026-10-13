# Feature Engineering Toolkit

A reproducible tabular-data preprocessing example built with scikit-learn. It demonstrates a leakage-safe train/test workflow using `Pipeline` and `ColumnTransformer`.

## Features

- Deterministic synthetic customer-like dataset for demonstration
- Train/test split before fitting preprocessing steps
- Median imputation and optional standardization for numeric columns
- Most-frequent imputation and one-hot encoding for categorical columns
- Unseen categories handled safely in the test set
- Reusable preprocessing pipeline with a downstream baseline estimator
- Unit tests for split behavior, transformed dimensions, and missing-value handling

> The included dataset is synthetic. It is not real customer, clinical, or research data.

## Requirements

Python 3.10+ is recommended.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Run the example

From the repository root:

```bash
python -m src.feature_engineering
```

The script prints dataset shape, split sizes, transformed feature count, and baseline test metrics. It writes a transformed test preview and feature-name list to `outputs/`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Leakage prevention

1. Split raw rows into train and test sets first.
2. Fit imputers, scalers, and encoders only through `pipeline.fit(X_train, y_train)`.
3. Transform test data only through `pipeline.predict(X_test)` or `pipeline.named_steps["preprocessor"].transform(X_test)`.
4. Do not compute imputation values, scaling parameters, or category vocabularies from the full dataset before splitting.

This repository is an educational baseline. Metrics are generated from synthetic data and must not be interpreted as evidence of real-world model performance.

## Structure

```text
feature-engineering-toolkit/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   └── feature_engineering.py
├── tests/
│   └── test_feature_engineering.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## License

MIT.
