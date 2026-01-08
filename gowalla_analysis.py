import pandas as pd

# ---- LOAD DATA ----
FILE_PATH = "loc-gowalla_totalCheckins.txt"
col_names = ["user_id", "timestamp", "latitude", "longitude", "location_id"]

df = pd.read_csv(
    FILE_PATH,
    sep=r"\s+",
    header=None,
    names=col_names,
    engine="python"
)

print("Loaded:", len(df), "rows (check-ins)")

# ---- BASIC DATASET STATISTICS ----
num_users = df["user_id"].nunique()
num_locations = df["location_id"].nunique()
num_checkins = len(df)

print("\n=== GOWALLA DATASET STATISTICS ===")
print(f"Users:        {num_users:,}")
print(f"Locations:    {num_locations:,}")
print(f"Check-ins:    {num_checkins:,}")

# ---- OPTIONAL: TOP LAT/LON FREQUENCIES (if you still want them) ----
df["lat_round"] = df["latitude"].round(4)
df["lon_round"] = df["longitude"].round(4)

top_locations = (
    df.groupby(["lat_round", "lon_round"])
      .size()
      .reset_index(name="count")
      .sort_values("count", ascending=False)
      .head(20)
)

print("\nTop 20 (lat, lon) combinations:")
print(top_locations)
