#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient-wise train/val/test split for MIMIC features (no leakage).
- Group by subject_id so a patient never appears across splits.
- 70/15/15 default proportions.
- Optional 'soft' stratification by admission_type.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random


def find_repo_root(start: Path = Path.cwd()) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start

def setup_logger(log_path: Path | None = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("split")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(level); ch.setFormatter(fmt); logger.addHandler(ch)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(level); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def soft_stratified_group_split(subjects: List[int],
                                subject_label: Dict[int, str],
                                ratios=(0.70, 0.15, 0.15),
                                seed=42) -> Tuple[set, set, set]:
    """
    Greedy bin-packing to keep label distribution similar across splits, grouped by subject.
    Labels are coarse (e.g., admission_type). Falls back to uniform if labels missing.
    """
    rnd = random.Random(seed)
    subs = list(subjects)
    rnd.shuffle(subs)

    target_sizes = [int(len(subs) * r) for r in ratios]
    # make sure totals add up
    while sum(target_sizes) < len(subs): target_sizes[0] += 1

    splits = [set(), set(), set()]
    label_counts = [Counter(), Counter(), Counter()]

    for s in subs:
        lab = subject_label.get(s, "UNKNOWN")
        # choose split with (1) available capacity and (2) minimal label imbalance
        best_i, best_score = None, None
        for i in range(3):
            if len(splits[i]) >= target_sizes[i]:  # capacity check
                continue
            # score = current label proportion deviation if we add to split i
            total_i = max(1, len(splits[i]) + 1)
            lab_prop = (label_counts[i][lab] + 1) / total_i
            # prefer closer to global prior (unknown here), so just minimize lab_prop drift from others
            # simpler: choose the split with the *lowest* lab_prop to spread label
            score = lab_prop
            if (best_score is None) or (score < best_score):
                best_score = score
                best_i = i
        if best_i is None:
            # all full due to rounding — put in smallest split
            best_i = min(range(3), key=lambda j: len(splits[j]))
        splits[best_i].add(s)
        label_counts[best_i][lab] += 1

    return splits[0], splits[1], splits[2]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patient-wise train/val/test split (no leakage).")
    p.add_argument("--input", type=str, help="Features CSV (default: <repo>/data/processed/mimic_features.csv)")
    p.add_argument("--out_dir", type=str, help="Output dir (default: <repo>/data/processed)")
    p.add_argument("--log", type=str, help="Log file (default: <repo>/logs/split.log)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--ratios", type=str, default="0.70,0.15,0.15", help="Comma ratios for train,val,test")
    p.add_argument("--soft_stratify", action="store_true", help="Approximate stratification by admission_type")
    return p.parse_args()

def main() -> None:
    repo = find_repo_root()
    args = parse_args()
    input_path = Path(args.input) if args.input else repo / "data" / "processed" / "mimic_features.csv"
    out_dir    = Path(args.out_dir) if args.out_dir else repo / "data" / "processed"
    log_path   = Path(args.log) if args.log else repo / "logs" / "split.log"

    logger = setup_logger(log_path)
    logger.info(f"Repo root: {repo}")
    logger.info(f"Input:     {input_path}")
    logger.info(f"Out dir:   {out_dir}")
    logger.info(f"Seed:      {args.seed}")
    ratios = tuple(float(x) for x in args.ratios.split(","))
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    logger.info(f"Ratios:    train/val/test = {ratios}")

    if not input_path.exists():
        raise FileNotFoundError(f"Features not found: {input_path}")

    df = pd.read_csv(input_path, encoding="utf-8")
    if "subject_id" not in df.columns:
        raise ValueError("subject_id column is required for group split.")
    subjects = df["subject_id"].astype(int).unique().tolist()
    logger.info(f"Unique subjects: {len(subjects)} | Rows: {len(df)}")

    # derive a coarse label per subject (admission_type mode), for soft stratification
    subj_label = defaultdict(lambda: "UNKNOWN")
    if args.soft_stratify:
        # try to reconstruct admission_type from OHE
        if "admission_type" in df.columns:
            adm_col = "admission_type"
        else:
            # infer from ohe
            adm_cols = [c for c in df.columns if c.startswith("adm_")]
            if adm_cols:
                # make a string label from highest one-hot index present
                tmp = df[["subject_id"] + adm_cols].copy()
                tmp["label"] = tmp[adm_cols].idxmax(axis=1).fillna("adm_OTHER")
                subj_label = tmp.groupby("subject_id")["label"].agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
            else:
                logger.warning("No admission_type or adm_* OHE found; soft stratification disabled.")
                args.soft_stratify = False
        if args.soft_stratify and "admission_type" in df.columns:
            tmp = df[["subject_id", "admission_type"]].astype({"subject_id": int, "admission_type": "string"})
            subj_label = tmp.groupby("subject_id")["admission_type"].agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()

    if args.soft_stratify:
        tr_subs, va_subs, te_subs = soft_stratified_group_split(subjects, subj_label, ratios=ratios, seed=args.seed)
    else:
        rnd = np.random.RandomState(args.seed)
        rnd.shuffle(subjects)
        n = len(subjects)
        n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
        tr_subs = set(subjects[:n_tr])
        va_subs = set(subjects[n_tr:n_tr+n_va])
        te_subs = set(subjects[n_tr+n_va:])

    def by_subject(sset: set) -> pd.DataFrame:
        return df[df["subject_id"].isin(sset)].copy()

    train_df, val_df, test_df = by_subject(tr_subs), by_subject(va_subs), by_subject(te_subs)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    def describe(split_df: pd.DataFrame, name: str) -> Dict:
        d = {"rows": int(len(split_df)), "subjects": int(split_df["subject_id"].nunique())}
        # summarize a few common categories if present
        for col in ["admission_type"]:
            if col in split_df.columns:
                vc = split_df[col].value_counts().to_dict()
                d[f"{col}_dist"] = vc
        # summarize OHE admission buckets if present
        adm_cols = [c for c in split_df.columns if c.startswith("adm_")]
        if adm_cols:
            d["adm_ohe_means"] = split_df[adm_cols].mean().round(4).to_dict()
        return d

    report = {
        "paths": {
            "input": str(input_path.resolve()),
            "train": str((out_dir / "train.csv").resolve()),
            "val": str((out_dir / "val.csv").resolve()),
            "test": str((out_dir / "test.csv").resolve()),
            "log": str(log_path.resolve()),
        },
        "counts": {
            "all_rows": int(len(df)),
            "all_subjects": int(len(subjects))
        },
        "splits": {
            "train": describe(train_df, "train"),
            "val": describe(val_df, "val"),
            "test": describe(test_df, "test"),
        },
        "ratios": ratios,
        "soft_stratify": args.soft_stratify,
    }

    report_path = repo / "logs" / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote split report → {report_path.resolve()}")
    logger.info("Done.")

if __name__ == "__main__":
    main()