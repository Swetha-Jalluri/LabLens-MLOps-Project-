#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 0 — repo root + paths (portable)
from pathlib import Path

def find_repo_root(start: Path = Path.cwd()):
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start

REPO = find_repo_root()
DATA_RAW = REPO / "data" / "raw"
DATA_PROCESSED = REPO / "data" / "processed"
LOGS = REPO / "logs"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(exist_ok=True)

RAW_CSV = DATA_RAW / "mimic_discharge_with_demographics.csv"
CLEAN_CSV = DATA_PROCESSED / "mimic_clean_step2.csv"  # intermediate output

print("Repo:", REPO)
print("Raw exists:", RAW_CSV.exists(), f"→ {RAW_CSV}")
print("Processed dir:", DATA_PROCESSED)
print("Output (intermediate):", CLEAN_CSV)


# In[2]:


# Cell 1 — load raw CSV
import pandas as pd

df = pd.read_csv(
    RAW_CSV,
    encoding="utf-8",
    dtype_backend="numpy_nullable",
    keep_default_na=True
)

print("Loaded shape:", df.shape)
print("Columns:", list(df.columns))
df.head(2)


# In[3]:


# Cell 2A — normalize text + basic filters
import re

before_rows = len(df)

# 1) ensure column present
assert "cleaned_text" in df.columns, "cleaned_text column not found"

# 2) normalization helpers
def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("\x00", "")                 # strip stray nulls
    s = re.sub(r"\s+", " ", s)                # collapse whitespace/newlines
    return s.strip()

df["cleaned_text"] = df["cleaned_text"].astype("string").fillna("")
df["cleaned_text"] = df["cleaned_text"].map(normalize_text)

# 3) length features
df["text_chars"] = df["cleaned_text"].str.len()
df["text_tokens"] = df["cleaned_text"].str.split().map(len)

# 4) drop rows with empty/very short notes (keep conservative threshold)
min_chars = 50
mask_keep = df["text_chars"] >= min_chars
dropped_short = int((~mask_keep).sum())
df = df.loc[mask_keep].copy()

# 5) enforce uniqueness by hadm_id (should already be unique)
dups = int(df["hadm_id"].duplicated().sum())
if dups:
    df = df.drop_duplicates(subset=["hadm_id"], keep="first").copy()

after_rows = len(df)

print("=== STEP 2A SUMMARY ===")
print(f"Rows before: {before_rows}")
print(f"Dropped too-short/empty: {dropped_short}")
print(f"Duplicate hadm_id removed: {dups}")
print(f"Rows after: {after_rows}")
print("Columns now:", list(df.columns)[:8], "... (+ length features)")


# In[4]:


# Cell 2B — save intermediate cleaned file
from pathlib import Path

CLEAN_STEP2 = (DATA_PROCESSED / "mimic_clean_step2.csv")
df.to_csv(
    CLEAN_STEP2,
    index=False,
    encoding="utf-8",
    quoting=1,          # QUOTE_ALL (robust for commas/newlines in text)
    lineterminator="\n",
    escapechar="\\"
)
print(f"Saved intermediate → {CLEAN_STEP2.resolve()}")


# In[5]:


# Cell 3A — schema tidy + light NA handling

# 1) desired column order (keeps demographics for bias checks)
cols = [
    "subject_id", "hadm_id",
    "gender", "ethnicity", "insurance", "language", "marital_status",
    "admission_type", "age_at_admission",
    "cleaned_text",
    "text_chars", "text_tokens",
    "lab_summary", "total_labs", "abnormal_lab_count",
    "diagnosis_count", "top_diagnoses"
]

# keep only columns that exist (robust)
cols_exist = [c for c in cols if c in df.columns]
df = df[cols_exist].copy()

# 2) enforce dtypes where reasonable (nullable ints stay nullable)
int_like = ["subject_id", "hadm_id", "age_at_admission",
            "total_labs", "abnormal_lab_count", "diagnosis_count", "text_chars", "text_tokens"]
for c in int_like:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# 3) light NA policy:
#    - don't drop rows here (we'll evaluate bias later)
#    - fill language with 'UNKNOWN' (common in MIMIC)
if "language" in df.columns:
    df["language"] = df["language"].fillna("UNKNOWN")

# 4) sanity checks
required_min = ["hadm_id", "cleaned_text"]
missing_required = {c: int(df[c].isna().sum()) for c in required_min if c in df.columns}
print("Missing required fields:", missing_required)

print("\nFinal column order:")
print(df.columns.tolist())

print("\nPreview:")
df.head(2)


# In[6]:


# Cell 3B — save final processed CSV
FINAL_CSV = (DATA_PROCESSED / "mimic_cleaned.csv")

df.to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8",
    quoting=1,          # QUOTE_ALL
    lineterminator="\n",
    escapechar="\\"
)

print("=== PREPROCESSING COMPLETE ===")
print(f"Saved final → {FINAL_CSV.resolve()}")
print(f"Rows: {len(df)}  |  Cols: {len(df.columns)}")

# quick QA counts
if "gender" in df.columns:
    print("\nGender counts (top):")
    print(df["gender"].value_counts().head(5).to_string())

if "ethnicity" in df.columns:
    print("\nEthnicity counts (top):")
    print(df["ethnicity"].value_counts().head(5).to_string())

if "admission_type" in df.columns:
    print("\nAdmission type counts:")
    print(df["admission_type"].value_counts().to_string())


# In[7]:


# Cell 4 — lightweight validation checks

issues = []

# ─────────── REQUIRED COLUMNS ───────────
required = ["hadm_id", "cleaned_text"]
for col in required:
    if col not in df.columns:
        issues.append(f"❌ Missing required column: {col}")

# ─────────── NO EMPTY TEXT ───────────
if (df["cleaned_text"].str.len() < 20).any():
    issues.append("⚠️ Some notes look suspiciously short (<20 chars).")

# ─────────── UNIQUE ADMISSIONS ───────────
if df["hadm_id"].duplicated().any():
    dup = df["hadm_id"].duplicated().sum()
    issues.append(f"⚠️ Duplicate hadm_id count: {dup}")

# ─────────── DEMOGRAPHIC VALIDITY ───────────
if "age_at_admission" in df.columns:
    if df["age_at_admission"].min() < 0:
        issues.append("❌ Negative ages detected")
    if df["age_at_admission"].max() > 110:
        issues.append("⚠️ Extremely old patients (check deidentification bucket)")

# ─────────── TEXT TOKEN HEALTH ───────────
if df["text_tokens"].mean() < 50:
    issues.append("⚠️ Token count mean too low (unexpected for discharge summaries)")

# ─────────── LAB SUMMARY HEALTH ───────────
if "lab_summary" in df.columns:
    if df["lab_summary"].isna().mean() > 0.15:
        issues.append("⚠️ Too many missing lab summaries (>15%)")

# ─────────── REPORT ───────────
print("=== VALIDATION SUMMARY ===")
if not issues:
    print("✅ All basic checks passed. Data looks healthy!")
else:
    for i in issues:
        print(i)

print("\nShape:", df.shape)
print("Preview:")
df.head(2)

