#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# configure seaborn
sns.set(style="whitegrid")

# detect repo root
REPO = Path.cwd().parents[0] if "notebooks" in str(Path.cwd()) else Path.cwd()
DATA_PROCESSED = REPO / "data" / "processed" / "mimic_cleaned.csv"

print("Repo root:", REPO)
print("Data:", DATA_PROCESSED)


# In[2]:


df = pd.read_csv(DATA_PROCESSED)

print("Loaded shape:", df.shape)
df.head()


# In[3]:


# === Bias Plot 1: Text length by gender ===
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
PLOTS_DIR = REPO / "logs" / "bias_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = PLOTS_DIR / "text_length_by_gender.png"

ycol = "cleaned_length" if "cleaned_length" in df.columns else "text_chars"

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="gender", y=ycol)
plt.title("Discharge Summary Length by Gender")
plt.xlabel("Gender")
plt.ylabel("Characters (cleaned text)")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()

print(f"✅ Saved plot -> {out_path}")


# In[4]:


# === Bias Plot 2: Text length by ethnicity ===

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = REPO / "logs" / "bias_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = PLOTS_DIR / "text_length_by_ethnicity.png"

# keep only top 8 ethnicities for readability
top_eth = df["ethnicity"].value_counts().index[:8]
subset = df[df["ethnicity"].isin(top_eth)]

ycol = "cleaned_length" if "cleaned_length" in df.columns else "text_chars"

plt.figure(figsize=(10,6))
sns.boxplot(data=subset, x="ethnicity", y=ycol)
plt.title("Discharge Summary Length by Ethnicity")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Ethnicity")
plt.ylabel("Characters (cleaned text)")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()

print(f"✅ Saved plot -> {out_path.resolve()}")


# In[5]:


# === Bias Plot 3: Abnormal lab counts by Age ===

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = REPO / "logs" / "bias_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = PLOTS_DIR / "abnormal_labs_by_age.png"

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="age_at_admission", y="abnormal_lab_count", alpha=0.4)
sns.regplot(data=df, x="age_at_admission", y="abnormal_lab_count", scatter=False, color="red")

plt.title("Abnormal Lab Count vs Age at Admission")
plt.xlabel("Age at admission")
plt.ylabel("Abnormal lab count")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.show()

print(f"✅ Saved plot -> {out_path.resolve()}")


# In[ ]:




