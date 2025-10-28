# data-pipeline/tests/test_preprocessing.py
import pandas as pd
import pytest
from conftest import get_callable

def test_import_preprocessing_module():
    mod = __import__("data_preprocessing.preprocessing", fromlist=["*"])
    assert mod is not None

def test_preprocess_returns_df(sample_df):
    mod = __import__("data_preprocessing.preprocessing", fromlist=["*"])
    fn = get_callable(mod, ["preprocess", "clean", "transform", "run"])
    if fn is None:
        pytest.skip("No preprocessing function implemented yet.")

    out = fn(sample_df.copy())
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(sample_df)

    if "text" in out:
        # typical expectation: missing text handled
        assert out["text"].isna().sum() == 0
    if "cleaned_length" in out:
        assert (out["cleaned_length"] >= 0).all()
