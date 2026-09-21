import numpy as np
import pandas as pd
import pytest
from src.pipeline import (
    calculate_feature_metrics,
    load_and_preprocess_dataset,
    scale_features,
)


def test_load_and_preprocess_dataset():
    X, y = load_and_preprocess_dataset()
    assert not X.empty
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0


def test_calculate_feature_metrics():
    X, _ = load_and_preprocess_dataset()
    metrics = calculate_feature_metrics(X)
    assert isinstance(metrics, pd.DataFrame)
    assert "mean" in metrics.columns
    assert "std" in metrics.columns


def test_scale_features():
    X, _ = load_and_preprocess_dataset()
    scaled_X = scale_features(X)
    assert isinstance(scaled_X, pd.DataFrame)
    assert scaled_X.shape == X.shape
    # Media debe ser aproximadamente 0 y desviación estándar poblacional 1
    assert np.allclose(scaled_X.mean(), 0, atol=1e-7)
    assert np.allclose(scaled_X.std(ddof=0), 1, atol=1e-7)