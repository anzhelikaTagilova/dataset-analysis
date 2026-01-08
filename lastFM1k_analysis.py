import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIGURATION
# =========================
DATA_FILE = "userid-timestamp-artid-artname-traid-traname.tsv"
CHUNKSIZE = 1_000_000
MIN_VALID_YEAR = 2002

# =========================
# TRACKERS
# =========================
min_dt = None
max_dt = None
total_rows = 0
valid_rows = 0
invalid_year_rows = 0
year_counts = {}

print("=== STREAMING DATASET ===")

# =========================
# STREAM SAFELY (ONLY NEEDED COLUMNS)
# =========================
for chunk in pd.read_csv(
    DATA_FILE,
    sep="\t",
    header=None,
    usecols=[0, 1],          #  only user_id and timestamp
    names=["user_id", "timestamp"],
    engine="python",
    quoting=3,              # csv.QUOTE_NONE
    on_bad_lines="skip",
    chunksize=CHUNKSIZE
):
    total_rows += len(chunk)

    # Parse timestamps
    chunk["datetime"] = pd.to_datetime(
        chunk["timestamp"],
        format="%Y-%m-%dT%H:%M:%SZ",
        errors="coerce"
    )

    # Drop invalid timestamps
    chunk = chunk.dropna(subset=["datetime"])
    valid_rows += len(chunk)

    # Update min/max
    cmin = chunk["datetime"].min()
    cmax = chunk["datetime"].max()

    min_dt = cmin if min_dt is None else min(min_dt, cmin)
    max_dt = cmax if max_dt is None else max(max_dt, cmax)

    # Year stats
    years = chunk["datetime"].dt.year
    for y, cnt in years.value_counts().items():
        year_counts[y] = year_counts.get(y, 0) + cnt

    invalid_year_rows += (years < MIN_VALID_YEAR).sum()

print("\n=== DATASET STATS ===")
print(f"Total rows processed      : {total_rows:,}")
print(f"Rows with valid timestamp : {valid_rows:,}")
print(f"Earliest datetime         : {min_dt}")
print(f"Latest datetime           : {max_dt}")
print(f"Time span                 : {(max_dt - min_dt).days} days "
      f"(~{(max_dt - min_dt).days / 365:.2f} years)")
print(f"Rows before {MIN_VALID_YEAR} : {invalid_year_rows:,} "
      f"({invalid_year_rows / valid_rows * 100:.6f}%)\n")

# =========================
# YEAR DISTRIBUTION
# =========================
year_series = pd.Series(year_counts).sort_index()

print("=== YEAR DISTRIBUTION ===")
print(year_series.head(10))
print("...\n")

# =========================
# PLOT
# =========================
plt.figure(figsize=(11, 4))
year_series.plot(kind="bar")
plt.axvline(
    x=list(year_series.index).index(MIN_VALID_YEAR),
    color="red",
    linestyle="--",
    label="Last.fm launch (2002)"
)
plt.title("Last.fm Listening Events by Year")
plt.xlabel("Year")
plt.ylabel("Number of events")
plt.legend()
plt.tight_layout()
plt.show()
