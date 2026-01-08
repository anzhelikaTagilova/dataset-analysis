import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "."  # project folder
TIMESTAMP_FILE = "user_taggedartists-timestamps.dat"
DATEPART_FILE = "user_taggedartists.dat"
MIN_VALID_YEAR = 2002  # Last.fm launched in 2002

# =========================
# 1. LOAD TIMESTAMP DATA
# =========================
ts_path = os.path.join(DATA_DIR, TIMESTAMP_FILE)

if not os.path.exists(ts_path):
    raise FileNotFoundError(f"{TIMESTAMP_FILE} not found in project folder")

df_ts = pd.read_csv(ts_path, sep="\t")

# Convert Unix timestamp (milliseconds → datetime)
df_ts["datetime"] = pd.to_datetime(df_ts["timestamp"], unit="ms", errors="coerce")

print("=== TIMESTAMP FILE LOADED ===")
print(f"Total records: {len(df_ts)}\n")

# =========================
# 2. BASIC TIME SPAN
# =========================
min_dt = df_ts["datetime"].min()
max_dt = df_ts["datetime"].max()

print("=== RAW TIME SPAN ===")
print(f"Earliest datetime: {min_dt}")
print(f"Latest datetime  : {max_dt}")
print(f"Total span       : {(max_dt - min_dt).days} days\n")

# =========================
# 3. YEAR DISTRIBUTION
# =========================
df_ts["year"] = df_ts["datetime"].dt.year
year_counts = df_ts["year"].value_counts().sort_index()

print("=== YEAR DISTRIBUTION ===")
print(year_counts.head(10))
print("...")
print(year_counts.tail(10), "\n")

# =========================
# 4. DETECT OUTLIERS
# =========================
outliers = df_ts[df_ts["year"] < MIN_VALID_YEAR]

print("=== OUTLIER ANALYSIS ===")
print(f"Records before {MIN_VALID_YEAR}: {len(outliers)}")
print(f"Percentage of dataset: {len(outliers) / len(df_ts) * 100:.6f}%\n")

if len(outliers) > 0:
    print("Sample outliers:")
    print(outliers[["userID", "artistID", "datetime"]].head(5), "\n")

# =========================
# 5. CLEAN DATASET
# =========================
df_clean = df_ts[df_ts["year"] >= MIN_VALID_YEAR].copy()

min_clean = df_clean["datetime"].min()
max_clean = df_clean["datetime"].max()

print("=== CLEANED DATASET ===")
print(f"Earliest valid datetime: {min_clean}")
print(f"Latest valid datetime  : {max_clean}")
print(f"Cleaned time span      : {(max_clean - min_clean).days} days "
      f"(~{(max_clean - min_clean).days / 365:.2f} years)")
print(f"Removed records        : {len(df_ts) - len(df_clean)}\n")

# =========================
# 6. OPTIONAL: SAVE CLEANED FILE
# =========================
OUTPUT_FILE = "user_taggedartists-timestamps_cleaned.dat"
df_clean.drop(columns=["year"]).to_csv(
    OUTPUT_FILE, sep="\t", index=False
)

print(f"Cleaned file saved as: {OUTPUT_FILE}\n")

# =========================
# 7. VISUALIZATION
# =========================
plt.figure(figsize=(10, 4))
year_counts.plot(kind="bar")
plt.axvline(MIN_VALID_YEAR - 0.5, color="red", linestyle="--", label="Last.fm launch")
plt.title("Distribution of Last.fm Timestamps by Year")
plt.xlabel("Year")
plt.ylabel("Number of interactions")
plt.legend()
plt.tight_layout()
plt.show()
