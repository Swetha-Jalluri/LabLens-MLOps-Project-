# data-pipeline/tests/conftest.py
import sys
from pathlib import Path
import pandas as pd
import pytest

# Make src/ importable
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

@pytest.fixture
def tmp_repo(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path

@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "hadm_id": [11, 22, 33, 44, 55],
            "text": [
                "Patient discharged in stable condition.",
                "No known allergies.",
                None,
                "Follow-up in 2 weeks. Re-check labs.",
                "Fever resolved; hydration advised.",
            ],
            "gender": ["M", "F", "F", "M", "F"],
            "ethnicity": ["WHITE", "BLACK", "ASIAN", "HISPANIC", "WHITE"],
            "age_at_admission": [25, 41, 67, 72, 54],
        }
    )

def get_callable(mod, names):
    """Return the first callable found in `names` from a module, else None."""
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None
