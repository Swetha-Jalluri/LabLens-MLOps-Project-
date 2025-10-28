#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack model-ready datasets for MIMIC-III summarization.

What it does
------------
1) Loads split feature CSVs (train/val/test) from data/processed/.
2) Adds `cleaned_text` by joining with data/processed/mimic_cleaned.csv on hadm_id.
3) Selects numeric feature columns (excludes ids/text).
4) Computes z-score stats on TRAIN ONLY and standardizes train/val/test.
5) Saves:
   - data/model/train.csv, val.csv, test.csv (ids, cleaned_text, standardized features)
   - logs/feature_spec.json (ordered feature list)
   - logs/scaler_stats.json (per-feature mean/std)

Notes
-----
- No sklearn dependency; means/stds computed in pure pandas/numpy.
- Optionally write Parquet via --parquet (requires pyarrow; falls back to CSV if unavailable).
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ------------------------- utils -------------------------

def find_repo_root(start: Path = Path.cwd()) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start

def setup_logger(log_path: Path | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("pack_model_data")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def load_split(name: str, processed_dir: Path) -> pd.DataFrame:
    p = processed_dir / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing split CSV: {p}")
    return pd.read_csv(p, encoding="utf-8")

def standardize(df: pd.DataFrame, means: Dict[str, float], stds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in means.keys():
        mu = means[c]
        sd = stds[c]
        if sd == 0 or np.isnan(sd):
            out[c] = 0.0
        else:
            out[c] = (out[c] - mu) / sd
    return out

def try_write_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        import pyarrow  # noqa: F401
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


# ------------------------- main pack logic -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create model-ready packs (text + standardized features).")
    p.add_argument("--processed_dir", type=str, help="Dir with processed CSVs (default: <repo>/data/processed)")
    p.add_argument("--out_dir", type=str, help="Output dir (default: <repo>/data/model)")
    p.add_argument("--log", type=str, help="Log file (default: <repo>/logs/pack_model_data.log)")
    p.add_argument("--parquet", action="store_true", help="Also write Parquet files (requires pyarrow).")
    return p.parse_args()

def main() -> None:
    repo = find_repo_root()
    args = parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else (repo / "data" / "processed")
    out_dir       = Path(args.out_dir) if args.out_dir else (repo / "data" / "model")
    log_path      = Path(args.log) if args.log else (repo / "logs" / "pack_model_data.log")

    logger = setup_logger(log_path)
    logger.info(f"Repo root:      {repo}")
    logger.info(f"Processed dir:  {processed_dir}")
    logger.info(f"Output dir:     {out_dir}")
    logger.info(f"Parquet:        {args.parquet}")

    # Load splits (engineered features)
    df_tr = load_split("train", processed_dir)
    df_va = load_split("val", processed_dir)
    df_te = load_split("test", processed_dir)
    logger.info(f"Loaded splits: train={df_tr.shape} val={df_va.shape} test={df_te.shape}")

    # Load cleaned text (from preprocessing step)
    cleaned_path = processed_dir / "mimic_cleaned.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned text file: {cleaned_path}")
    df_text = pd.read_csv(cleaned_path, encoding="utf-8", usecols=["hadm_id", "cleaned_text"])
    logger.info(f"Loaded cleaned_text: {df_text.shape}")

    # Join text into splits (by hadm_id)
    def add_text(d: pd.DataFrame) -> pd.DataFrame:
        if "hadm_id" not in d.columns:
            raise ValueError("hadm_id required in feature splits.")
        return d.merge(df_text, on="hadm_id", how="left")

    df_tr = add_text(df_tr)
    df_va = add_text(df_va)
    df_te = add_text(df_te)

    # Identify feature columns (numeric only), exclude identifiers & text
    id_cols = [c for c in ["subject_id", "hadm_id"] if c in df_tr.columns]
    drop_cols = set(id_cols + ["cleaned_text"])
    num_tr = df_tr.select_dtypes(include=["number"]).copy()
    feat_cols = [c for c in num_tr.columns if c not in drop_cols]

    if not feat_cols:
        raise RuntimeError("No numeric feature columns found for modeling.")

    # Ensure consistent columns across splits
    for name, d in [("val", df_va), ("test", df_te)]:
        missing = set(feat_cols) - set(d.columns)
        if missing:
            raise RuntimeError(f"{name} split missing feature columns: {sorted(missing)[:5]}...")

    # Compute TRAIN means/stds and standardize all splits
    means = {c: float(pd.to_numeric(df_tr[c], errors="coerce").fillna(0).mean()) for c in feat_cols}
    stds  = {c: float(pd.to_numeric(df_tr[c], errors="coerce").fillna(0).std(ddof=0)) for c in feat_cols}

    def cast_numeric(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        for c in feat_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        return out

    df_tr = cast_numeric(df_tr)
    df_va = cast_numeric(df_va)
    df_te = cast_numeric(df_te)

    df_tr[feat_cols] = standardize(df_tr[feat_cols], means, stds)
    df_va[feat_cols] = standardize(df_va[feat_cols], means, stds)
    df_te[feat_cols] = standardize(df_te[feat_cols], means, stds)

    # Final column order: ids, text, features (stable order)
    ordered = id_cols + ["cleaned_text"] + feat_cols
    df_tr = df_tr[ordered].copy()
    df_va = df_va[ordered].copy()
    df_te = df_te[ordered].copy()

    # Save CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {
        "train": out_dir / "train.csv",
        "val":   out_dir / "val.csv",
        "test":  out_dir / "test.csv",
    }
    for name, d in [("train", df_tr), ("val", df_va), ("test", df_te)]:
        d.to_csv(csv_paths[name], index=False, encoding="utf-8", lineterminator="\n")
    logger.info(f"Saved model CSVs → {out_dir}")

    # Optionally save Parquet too
    if args.parquet:
        pq_ok = True
        pq_ok &= try_write_parquet(df_tr, out_dir / "train.parquet")
        pq_ok &= try_write_parquet(df_va, out_dir / "val.parquet")
        pq_ok &= try_write_parquet(df_te, out_dir / "test.parquet")
        if pq_ok:
            logger.info("Saved Parquet files.")
        else:
            logger.warning("Parquet not written (pyarrow missing). CSVs are available.")

    # Save specs and scaler stats
    logs_dir = repo / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    spec_path = logs_dir / "feature_spec.json"
    stats_path = logs_dir / "scaler_stats.json"

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump({"feature_order": feat_cols}, f, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"mean": means, "std": stds}, f, indent=2)

    logger.info(f"Wrote feature spec → {spec_path.resolve()}")
    logger.info(f"Wrote scaler stats → {stats_path.resolve()}")
    logger.info("Pack complete.")


if __name__ == "__main__":
    main()
