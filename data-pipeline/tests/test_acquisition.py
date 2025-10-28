# data-pipeline/tests/test_acquisition.py
import pandas as pd
import pytest
from conftest import get_callable

def test_import_acquisition_module():
    mod = __import__("data_acquisition.acquisition", fromlist=["*"])
    assert mod is not None

def test_load_csv(tmp_repo):
    mod = __import__("data_acquisition.acquisition", fromlist=["*"])
    loader = get_callable(mod, ["load_raw", "load_csv", "read_raw", "read_source"])
    if loader is None:
        pytest.skip("No acquisition loader implemented yet.")

    csv_path = tmp_repo / "data" / "raw" / "tiny.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv_path, index=False)

    df = loader(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2

def test_main_smoke(tmp_repo, monkeypatch):
    mod = __import__("data_acquisition.acquisition", fromlist=["*"])
    main = get_callable(mod, ["main", "run"])
    if main is None:
        pytest.skip("No acquisition main()/run() implemented yet.")
    monkeypatch.setenv("LABLENS_DATA_DIR", str(tmp_repo / "data"))
    main()  # should not crash
