import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_engineering import (
    build_model_pipeline,
    build_preprocessor,
    make_synthetic_dataset,
)


class TestFeatureEngineering(unittest.TestCase):
    def test_synthetic_data_has_expected_rows_and_missing_values(self):
        X, y = make_synthetic_dataset(n_rows=100, seed=3)
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(len(y), 100)
        self.assertTrue(X.isna().any().any())

    def test_preprocessor_fits_and_transforms_with_missing_values(self):
        X, y = make_synthetic_dataset(n_rows=120, seed=4)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.25, random_state=5, stratify=y
        )
        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(X_train)
        transformed_test = preprocessor.transform(X_test)
        self.assertEqual(transformed_train.shape[1], transformed_test.shape[1])
        self.assertTrue(np.isfinite(transformed_train).all())
        self.assertTrue(np.isfinite(transformed_test).all())

    def test_model_pipeline_predicts_test_rows(self):
        X, y = make_synthetic_dataset(n_rows=150, seed=7)
        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=8, stratify=y
        )
        model = build_model_pipeline()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_unknown_category_is_handled(self):
        X_train = pd.DataFrame({
            "age": [20, 30, 40, np.nan, 50, 60],
            "annual_income": [20_000, 30_000, 40_000, 50_000, np.nan, 60_000],
            "visits_per_month": [1, 2, 3, 4, 5, 6],
            "membership": ["basic", "plus", "premium", "basic", "plus", "premium"],
            "region": ["north", "south", "east", "west", "north", "south"],
        })
        X_test = pd.DataFrame({
            "age": [35.0],
            "annual_income": [42_000.0],
            "visits_per_month": [3],
            "membership": ["vip"],
            "region": ["central"],
        })
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        transformed = preprocessor.transform(X_test)
        self.assertEqual(transformed.shape[0], 1)
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
