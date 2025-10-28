#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validation for MIMIC-III engineered features.
- Verifies schema presence, ranges, uniqueness, OHE integrity, and missingness.
- Writes a JSON summary + CSV of row-level issues (if any).
- Optional --strict flag makes the script exit non-zero if critical checks fail.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ───────────────────────── utils ─────────────────────────

def find_repo_root(start: Path = Path.cwd()) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start

def setup_logger(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("validation")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ───────────────────────── checks ─────────────────────────

CRITICAL_COLUMNS = [
    "subject_id", "hadm_id",
    "text_chars", "text_tokens", "sentences", "chars_per_token",
    "kw_chronic_disease", "kw_symptoms", "kw_medications", "kw_med_suffix_hits",
    "abnormal_lab_ratio", "diagnosis_unique_count", "comorbidity_score",
]

RANGE_CHECKS = {
    # col: (min, max, inclusive_min, inclusive_max)
    "text_chars": (1, 2_000_000, True, True),
    "text_tokens": (1, 300_000, True, True),
    "sentences": (0, 100_000, True, True),
    "chars_per_token": (0.5, 20.0, True, True),
    "abnormal_lab_ratio": (0.0, 1.0, True, True),
    "diagnosis_unique_count": (0, 10_000, True, True),
    "comorbidity_score": (0, 10_000, True, True),
}

def get_ohe_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("gender_", "eth_", "ins_", "adm_", "lang_")
    return [c for c in df.columns if c.startswith(prefixes)]

def check_presence(df: pd.DataFrame) -> Dict[str, List[str]]:
    missing = [c for c in CRITICAL_COLUMNS if c not in df.columns]
    return {"missing_columns": missing}

def check_uniqueness(df: pd.DataFrame) -> Dict[str, int]:
    dups = df["hadm_id"].duplicated().sum() if "hadm_id" in df.columns else -1
    return {"duplicate_hadm_id": int(dups)}

def check_ranges(df: pd.DataFrame) -> List[Tuple[str, int]]:
    problems = []
    for col, (mn, mx, inc_min, inc_max) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if inc_min:
            low = s < mn
        else:
            low = s <= mn
        if inc_max:
            high = s > mx
        else:
            high = s >= mx
        n_bad = int((low | high | s.isna()).sum()) if col != "sentences" else int(((low | high) | s.isna()).sum())
        # we allow NaN in some engineered ints? better to count them too:
        problems.append((col, n_bad))
    return problems

def check_ohe_integrity(df: pd.DataFrame) -> Dict[str, int]:
    ohe_cols = get_ohe_columns(df)
    non_binary = 0
    has_nan = 0
    for c in ohe_cols:
        s = df[c]
        non_binary += int(~s.isin([0, 1]).sum())
        has_nan += int(s.isna().sum())
    return {
        "ohe_cols": len(ohe_cols),
        "ohe_non_binary_count": non_binary,
        "ohe_nan_count": has_nan,
    }

def collect_row_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table of row-level violations for quick debugging."""
    issues = pd.DataFrame(index=df.index)

    # range rules
    for col, (mn, mx, inc_min, inc_max) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        low = s < mn if inc_min else s <= mn
        high = s > mx if inc_max else s >= mx
        bad = (low | high | s.isna())
        issues[f"{col}__bad"] = bad.astype("int8")

    # hadm_id uniqueness
    if "hadm_id" in df.columns:
        issues["hadm_id__duplicate"] = df["hadm_id"].duplicated(keep=False).astype("int8")

    # OHE binary + NaN
    for c in get_ohe_columns(df):
        s = df[c]
        non_binary = ~s.isin([0, 1])
        issues[f"{c}__nonbinary"] = non_binary.astype("int8")
        issues[f"{c}__nan"] = s.isna().astype("int8")

    # keep only rows with any issue
    any_issue = issues.sum(axis=1) > 0
    issues = issues.loc[any_issue]
    return issues.reset_index(drop=False).rename(columns={"index": "row"})


# ───────────────────────── CLI ─────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate engineered features CSV.")
    p.add_argument("--input", type=str, help="Path to features CSV (default: <repo>/data/processed/mimic_features.csv)")
    p.add_argument("--report", type=str, help="JSON report path (default: <repo>/logs/validation_report.json)")
    p.add_argument("--issues", type=str, help="CSV of row-level issues (default: <repo>/logs/validation_issues.csv)")
    p.add_argument("--log", type=str, help="Log file path (default: <repo>/logs/validation.log)")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if critical checks fail")
    return p.parse_args()


def main() -> None:
    repo = find_repo_root()
    default_input  = repo / "data" / "processed" / "mimic_features.csv"
    default_report = repo / "logs" / "validation_report.json"
    default_issues = repo / "logs" / "validation_issues.csv"
    default_log    = repo / "logs" / "validation.log"

    args = parse_args()
    input_path  = Path(args.input)  if args.input  else default_input
    report_path = Path(args.report) if args.report else default_report
    issues_path = Path(args.issues) if args.issues else default_issues
    log_path    = Path(args.log)    if args.log    else default_log

    logger = setup_logger(log_path)
    logger.info(f"Repo root: {repo}")
    logger.info(f"Input:     {input_path}")
    logger.info(f"Report:    {report_path}")
    logger.info(f"Issues:    {issues_path}")
    logger.info(f"Strict:    {args.strict}")

    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8")
    logger.info(f"Loaded features: rows={len(df)}, cols={len(df.columns)}")

    # presence
    presence = check_presence(df)
    missing = presence["missing_columns"]
    if missing:
        logger.error(f"Missing critical columns: {missing}")

    # uniqueness
    uniq = check_uniqueness(df)
    if uniq["duplicate_hadm_id"] > 0:
        logger.error(f"Duplicate hadm_id rows: {uniq['duplicate_hadm_id']}")

    # ranges
    range_problems = check_ranges(df)
    for col, nbad in range_problems:
        if nbad > 0:
            logger.warning(f"Range/NaN issues in {col}: {nbad}")

    # OHE integrity
    ohe = check_ohe_integrity(df)
    if ohe["ohe_non_binary_count"] > 0 or ohe["ohe_nan_count"] > 0:
        logger.warning(f"OHE integrity issues: non-binary={ohe['ohe_non_binary_count']} NaN={ohe['ohe_nan_count']} (across {ohe['ohe_cols']} OHE cols)")

    # row-level issues table
    issues_df = collect_row_issues(df)
    if len(issues_df) > 0:
        issues_path.parent.mkdir(parents=True, exist_ok=True)
        issues_df.to_csv(issues_path, index=False)
        logger.info(f"Wrote row-level issues CSV → {issues_path.resolve()}")
    else:
        logger.info("No row-level issues detected.")

    # summary JSON
    summary = {
        "shape": {"rows": int(len(df)), "cols": int(len(df.columns))},
        "presence": presence,
        "uniqueness": uniq,
        "range_issues": {col: int(nbad) for col, nbad in range_problems},
        "ohe": ohe,
        "paths": {
            "input": str(input_path.resolve()),
            "report": str(report_path.resolve()),
            "issues": str(issues_path.resolve()),
            "log": str(log_path.resolve()),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote validation report JSON → {report_path.resolve()}")

    # strict mode → fail on critical problems
    critical_fail = bool(missing) or (uniq["duplicate_hadm_id"] > 0)
    if args.strict and critical_fail:
        logger.error("STRICT mode: critical validation failed.")
        raise SystemExit(2)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
