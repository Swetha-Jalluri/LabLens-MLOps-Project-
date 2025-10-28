#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Cell 0: Portable Repo Root & Directory Setup
from pathlib import Path
import sys

# --- Helper: auto-locate repo root by searching upward for configs/data_config.yaml ---
def find_repo_root(start: Path = Path.cwd()):
    cur = start
    while cur != cur.parent:
        if (cur / "configs" / "data_config.yaml").exists():
            return cur
        cur = cur.parent
    return start  # fallback: current working directory

# Detect repo root regardless of where notebook is run
REPO = find_repo_root()
print(f"Repository root detected: {REPO}")

# Create necessary directories
LOGS_DIR = REPO / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DATA_DIR = REPO / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

# Paths to key files
CONFIG_PATH = REPO / "configs" / "data_config.yaml"
REQUIREMENTS_PATH = REPO / "requirements.txt"
ENV_PATH = REPO / ".env"

# Verify important files
print("\n✓ Directory setup:")
print(f"  Logs:     {LOGS_DIR}")
print(f"  Data/raw: {DATA_RAW_DIR}")

print("\n✓ File verification:")
print(f"  config.yaml exists:        {CONFIG_PATH.exists()}")
print(f"  requirements.txt exists:   {REQUIREMENTS_PATH.exists()}")
print(f"  .env exists:               {ENV_PATH.exists()}")

# Output paths for saving during acquisition
OUTPUT_CSV = DATA_RAW_DIR / "mimic_discharge_with_demographics.csv"
LOG_FILE = LOGS_DIR / "data_acquisition.log"

print(f"\n✓ Output artifacts will be saved to:")
print(f"  CSV output: {OUTPUT_CSV}")
print(f"  Log file:   {LOG_FILE}")
print("\n✅ Notebook environment successfully initialized!")


# In[2]:


# Cell 1: Import Libraries & Configure Logging
import logging
import os
from typing import Tuple, Optional
from pathlib import Path

from google.cloud import bigquery
from google.cloud.bigquery import ArrayQueryParameter
from google.oauth2.service_account import Credentials
import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with UTF-8 encoding (pathlib-based)
REPO = Path.cwd()
LOG_FILE = REPO / "logs" / "data_acquisition.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
print('[OK] Libraries imported and logging configured')


# In[8]:


# Cell 2: Improved MIMICDataAcquisition with all 5 fixes
from pathlib import Path

class MIMICDataAcquisition:
    """Improved data acquisition with portability and safety fixes."""

    def __init__(self, config_path: str = None, output_dir: Path = None):
        self.repo = Path.cwd()
        if config_path is None:
            config_path = self.repo / "configs" / "data_config.yaml"
        else:
            config_path = Path(config_path)

        if output_dir is None:
            output_dir = self.repo / "data" / "raw"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_path)
        self.client = self._authenticate_bigquery()
        logger.info("MIMIC Data Acquisition initialized successfully")

    def _load_config(self, config_path: Path) -> dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {config_path}")
            raise

    def _authenticate_bigquery(self) -> bigquery.Client:
        try:
            service_account_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if service_account_key and os.path.exists(service_account_key):
                credentials = Credentials.from_service_account_file(service_account_key)
                client = bigquery.Client(
                    project=self.config['gcp']['project_id'],
                    credentials=credentials
                )
                logger.info("[OK] Authenticated using service account")
            else:
                client = bigquery.Client(project=self.config['gcp']['project_id'])
                logger.info("[OK] Authenticated using application default credentials")

            client.query("SELECT 1").result()
            logger.info("[OK] BigQuery connection test successful")
            return client
        except Exception as e:
            logger.error(f"BigQuery authentication failed: {str(e)}")
            raise ValueError("Failed to authenticate with BigQuery") from e

    def fetch_discharge_summaries(self) -> pd.DataFrame:
        sample_limit = self.config['data_acquisition']['sample_limit']
        query = f"""
        WITH discharge AS (
            SELECT 
                n.subject_id,
                n.hadm_id,
                n.charttime,
                REGEXP_REPLACE(n.text, r'\\[\\*\\*[^\\]]*\\*\\*\\]', '') AS cleaned_text,
                LENGTH(n.text) AS original_length,
                LENGTH(REGEXP_REPLACE(n.text, r'\\[\\*\\*[^\\]]*\\*\\*\\]', '')) AS cleaned_length,
                ROW_NUMBER() OVER (PARTITION BY n.hadm_id ORDER BY LENGTH(n.text) DESC) AS rn
            FROM `{self.config['bigquery']['dataset_notes']}.noteevents` n
            WHERE n.category = 'Discharge summary'
        )
        SELECT 
            subject_id,
            hadm_id,
            charttime,
            cleaned_text,
            original_length,
            cleaned_length
        FROM discharge
        WHERE rn = 1
        QUALIFY ROW_NUMBER() OVER (ORDER BY hadm_id) <= {sample_limit}
        """

        try:
            logger.info("Fetching discharge summaries (dedupe first, then sample)...")
            df = self.client.query(query).to_dataframe()
            logger.info(f"[OK] Fetched {len(df)} unique discharge summaries")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch discharge summaries: {str(e)}")
            raise

    def fetch_demographics_and_admissions(self, discharge_hadm_ids: list) -> pd.DataFrame:
        query = f"""
        SELECT 
            p.subject_id,
            a.hadm_id,
            p.gender,
            a.ethnicity,
            a.insurance,
            a.language,
            a.marital_status,
            a.admission_type,
            a.admittime,
            CASE 
                WHEN DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) >= 89 THEN 89
                WHEN DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) < 0 THEN NULL
                ELSE DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR)
            END as age_at_admission
        FROM `{self.config['bigquery']['dataset_clinical']}.patients` p
        INNER JOIN `{self.config['bigquery']['dataset_clinical']}.admissions` a 
            ON p.subject_id = a.subject_id AND a.hadm_id IN UNNEST(@hadm_ids)
        """

        try:
            logger.info("Fetching demographics and admission data...")
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    ArrayQueryParameter("hadm_ids", "INT64", discharge_hadm_ids)
                ]
            )
            df = self.client.query(query, job_config=job_config).to_dataframe()
            logger.info(f"[OK] Fetched {len(df)} demographic records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch demographics: {str(e)}")
            raise

    def fetch_lab_results(self, discharge_hadm_ids: list) -> pd.DataFrame:
        query = f"""
        SELECT 
            le.hadm_id,
            le.itemid,
            d.label,
            le.value,
            le.valueuom,
            le.flag,
            le.charttime
        FROM `{self.config['bigquery']['dataset_clinical']}.labevents` le
        INNER JOIN `{self.config['bigquery']['dataset_clinical']}.d_labitems` d 
            ON le.itemid = d.itemid
        WHERE le.hadm_id IN UNNEST(@hadm_ids)
        """

        try:
            logger.info("Fetching lab results (scoped to discharge admissions)...")
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    ArrayQueryParameter("hadm_ids", "INT64", discharge_hadm_ids)
                ]
            )
            df = self.client.query(query, job_config=job_config).to_dataframe()
            logger.info(f"[OK] Fetched {len(df)} lab result records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch lab results: {str(e)}")
            raise

    def fetch_diagnoses(self, discharge_hadm_ids: list) -> pd.DataFrame:
        query = f"""
        SELECT 
            hadm_id,
            icd9_code,
            seq_num
        FROM `{self.config['bigquery']['dataset_clinical']}.diagnoses_icd`
        WHERE hadm_id IN UNNEST(@hadm_ids)
        """

        try:
            logger.info("Fetching diagnoses (scoped to discharge admissions)...")
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    ArrayQueryParameter("hadm_ids", "INT64", discharge_hadm_ids)
                ]
            )
            df = self.client.query(query, job_config=job_config).to_dataframe()
            logger.info(f"[OK] Fetched {len(df)} diagnosis records")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch diagnoses: {str(e)}")
            raise

    def aggregate_lab_results(self, labs_df: pd.DataFrame) -> pd.DataFrame:
        if labs_df.empty:
            logger.warning("No lab results to aggregate")
            return pd.DataFrame()

        agg_df = labs_df.groupby('hadm_id').agg({
            'label': lambda x: '; '.join(x.unique()[:10]),
            'flag': lambda x: (x == 'abnormal').sum(),
            'itemid': 'count'
        }).rename(columns={
            'label': 'lab_summary',
            'flag': 'abnormal_lab_count',
            'itemid': 'total_labs'
        }).reset_index()

        logger.info(f"[OK] Aggregated lab results to {len(agg_df)} admissions")
        return agg_df

    def aggregate_diagnoses(self, diagnoses_df: pd.DataFrame) -> pd.DataFrame:
        if diagnoses_df.empty:
            logger.warning("No diagnoses to aggregate")
            return pd.DataFrame()

        agg_df = diagnoses_df.groupby('hadm_id').agg({
            'icd9_code': lambda x: ', '.join([str(code) for code in x.unique() if code is not None][:5]),
            'seq_num': 'count'
        }).rename(columns={
            'icd9_code': 'top_diagnoses',
            'seq_num': 'diagnosis_count'
        }).reset_index()

        logger.info(f"[OK] Aggregated diagnoses to {len(agg_df)} admissions")
        return agg_df

    def merge_datasets(self, discharge_df, demographics_df, labs_agg_df, diagnoses_agg_df):
        logger.info("Merging all datasets...")

        merged = discharge_df.merge(
            demographics_df,
            on=['hadm_id', 'subject_id'],
            how='inner',
            validate='1:1'
        )
        logger.info(f"  After merge with demographics: {len(merged)} rows (1:1 validated)")

        if not labs_agg_df.empty:
            merged = merged.merge(
                labs_agg_df,
                on='hadm_id',
                how='left',
                validate='1:1'
            )
            logger.info(f"  After merge with labs: {len(merged)} rows (1:1 validated)")

        if not diagnoses_agg_df.empty:
            merged = merged.merge(
                diagnoses_agg_df,
                on='hadm_id',
                how='left',
                validate='1:1'
            )
            logger.info(f"  After merge with diagnoses: {len(merged)} rows (1:1 validated)")

        logger.info(f"[OK] Final merged dataset shape: {merged.shape}")
        return merged

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, dict]:
        logger.info("Running data validation...")

        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'warnings': [],
            'info': {}
        }

        if 'cleaned_text' in df:
            empty_texts = (df['cleaned_text'].str.len() == 0).sum()
            if empty_texts > 0:
                validation_results['warnings'].append(f"Found {empty_texts} empty discharge texts")

        if 'hadm_id' in df:
            duplicates = df['hadm_id'].duplicated().sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Found {duplicates} duplicate hadm_ids")

        is_valid = len(validation_results['warnings']) == 0

        if is_valid:
            logger.info("[OK] Data validation PASSED")
        else:
            logger.warning(f"[WARNING] Data validation found {len(validation_results['warnings'])} issues")

        return is_valid, validation_results

    def save_data(self, df: pd.DataFrame) -> Path:
        output_path = self.output_dir / "mimic_discharge_with_demographics.csv"

        try:
            df = df.drop(columns=[c for c in ["dob", "charttime"] if c in df.columns])

            df.to_csv(
                output_path,
                index=False,
                encoding='utf-8',
                quoting=1,
                lineterminator='\n'
            )

            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"[OK] Data saved")
            logger.info(f"  Path: {output_path.resolve()}")
            logger.info(f"  Size: {file_size_mb:.2f} MB")
            logger.info(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

            return output_path
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
            raise

    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, dict]:
        try:
            logger.info("="*60)
            logger.info("Starting MIMIC-III Data Acquisition Pipeline")
            logger.info("="*60)

            discharge_df = self.fetch_discharge_summaries()
            discharge_hadm_ids = discharge_df['hadm_id'].tolist()

            demographics_df = self.fetch_demographics_and_admissions(discharge_hadm_ids)
            labs_df = self.fetch_lab_results(discharge_hadm_ids)
            diagnoses_df = self.fetch_diagnoses(discharge_hadm_ids)

            labs_agg_df = self.aggregate_lab_results(labs_df)
            diagnoses_agg_df = self.aggregate_diagnoses(diagnoses_df)

            merged_df = self.merge_datasets(discharge_df, demographics_df, labs_agg_df, diagnoses_agg_df)

            is_valid, validation_results = self.validate_data(merged_df)

            output_path = self.save_data(merged_df)

            logger.info("="*60)
            logger.info("[OK] Pipeline completed successfully!")
            logger.info("="*60)

            return merged_df, validation_results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

print('[OK] MIMICDataAcquisition class defined')


# In[ ]:


# Cell 3: Run the Pipeline
try:
    acquisition = MIMICDataAcquisition()
    df, validation = acquisition.run_complete_pipeline()

    print("\n" + "="*60)
    print("[OK] DATA ACQUISITION COMPLETE")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nValidation Results:")
    print(f"  Total rows: {validation['total_rows']}")
    print(f"  Warnings: {len(validation['warnings'])}")

    if validation['warnings']:
        print(f"\n  Warnings:")
        for w in validation['warnings']:
            print(f"    - {w}")
    else:
        print(f"\n  ✓ No warnings!")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

except Exception as e:
    print(f"[ERROR] Error: {str(e)}")
    import traceback
    traceback.print_exc()


# In[10]:


# Cell 4: Explore the Data
print("="*60)
print("DATA STRUCTURE")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("DEMOGRAPHIC DISTRIBUTIONS (for bias detection)")
print("="*60)

print("\nGender distribution:")
print(df['gender'].value_counts())

print("\nEthnicity distribution (top 10):")
print(df['ethnicity'].value_counts().head(10))

print("\nAge at admission:")
print(f"  Min: {df['age_at_admission'].min()}")
print(f"  Max: {df['age_at_admission'].max()}")
print(f"  Mean: {df['age_at_admission'].mean():.1f}")
print(f"  Median: {df['age_at_admission'].median():.1f}")

print("\nAdmission type distribution:")
print(df['admission_type'].value_counts())

print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)
print(f"\nMissing values per column:")
print(df.isnull().sum())

print(f"\nUnique hadm_ids: {df['hadm_id'].nunique()}")
print(f"Total rows: {len(df)}")
print(f"Duplicates: {df['hadm_id'].duplicated().sum()}")

print("\n✓ Data exploration complete!")

