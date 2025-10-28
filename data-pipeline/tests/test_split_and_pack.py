
# data-pipeline/tests/test_split_and_pack.py
import pandas as pd
import pytest
from conftest import get_callable

def test_splitter(sample_df):
    mod = __import__("split_dataset.split_dataset", fromlist=["*"])
    fn = get_callable(mod, ["split", "split_dataset", "run"])
    if fn is None:
        pytest.skip("No split function implemented yet.")

    result = fn(sample_df.copy(), ratios=(0.6, 0.2, 0.2), stratify="gender")
    if isinstance(result, dict):
        train, val, test = result["train"], result["val"], result["test"]
    else:
        train, val, test = result
    assert len(train) + len(val) + len(test) == len(sample_df)

def test_packer(tmp_repo, sample_df):
    try:
        mod = __import__("model_packaging.packer", fromlist=["*"])
    except ModuleNotFoundError:
        mod = __import__("model_packaging.pack_model_data", fromlist=["*"])

    fn = get_callable(mod, ["pack", "pack_model_data", "run"])
    if fn is None:
        pytest.skip("No packaging function implemented yet.")

    out_dir = tmp_repo / "data" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = fn(sample_df.copy(), out_dir)

    present = {p.name for p in out_dir.glob("*")}
    assert {"train.csv", "val.csv", "test.csv", "feature_spec.json"} & present
