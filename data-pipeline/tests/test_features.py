# data-pipeline/tests/test_features.py
import pandas as pd
import pytest
from conftest import get_callable

def test_import_feature_module():
    mod = __import__("feature_engineering.feature_engineer", fromlist=["*"])
    assert mod is not None

def test_build_features(sample_df):
    mod = __import__("feature_engineering.feature_engineer", fromlist=["*"])
    fn = get_callable(mod, ["build_features", "engineer_features", "featurize", "run"])
    if fn is None:
        pytest.skip("No feature engineering function implemented yet.")

    out = fn(sample_df.copy())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(sample_df)

    # sanity for common numeric features if present
    for col in {"abnormal_lab_count", "num_sentences", "num_tokens"} & set(out.columns):
        assert (out[col] >= 0).all()
