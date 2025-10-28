# data-pipeline/tests/test_dag_parse.py
import importlib.util
from pathlib import Path
import pytest

airflow = pytest.importorskip("airflow")  # skip nicely if Airflow not installed

def test_dag_import_ok():
    repo = Path(__file__).resolve().parents[1]
    dag_file = repo / "dags" / "mimic_nlp_pipeline_dag.py"
    if not dag_file.exists() or dag_file.stat().st_size == 0:
        pytest.skip("DAG file missing or empty.")
    spec = importlib.util.spec_from_file_location("mimic_dag", dag_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "dag") or hasattr(mod, "dags")
