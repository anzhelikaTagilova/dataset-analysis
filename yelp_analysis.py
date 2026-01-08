import json
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

# ---------------------------
# Helpers
# ---------------------------

def read_jsonl(path: Path) -> pd.DataFrame:
    """Read Yelp JSON Lines file into a DataFrame."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def describe_distribution(series, name, bins=(1, 2, 5, 10, 20, 50, 100)):
    """Print useful distribution stats for a count series."""
    series = series.dropna()
    print(f"\n{name} distribution:")
    print(f"  count: {len(series)}")
    print(f"  mean: {series.mean():.2f}")
    print(f"  median: {series.median():.2f}")
    print(f"  min/max: {series.min()} / {series.max()}")
    for b in bins:
        pct = (series >= b).mean() * 100
        print(f"  % >= {b}: {pct:.2f}%")


def gini(values):
    """Gini coefficient for skewness/inequality."""
    values = np.array(values, dtype=float)
    if np.amin(values) < 0:
        values -= np.amin(values)
    values += 1e-9
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))


# ---------------------------
# Load data
# ---------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "yelp_dataset"

business_path = DATA_DIR / "yelp_academic_dataset_business.json"
checkin_path  = DATA_DIR / "yelp_academic_dataset_checkin.json"
review_path   = DATA_DIR / "yelp_academic_dataset_review.json"
tip_path      = DATA_DIR / "yelp_academic_dataset_tip.json"
user_path     = DATA_DIR / "yelp_academic_dataset_user.json"

print("Loading files...")

business_df = read_jsonl(business_path)
review_df   = read_jsonl(review_path)
tip_df      = read_jsonl(tip_path)
checkin_df  = read_jsonl(checkin_path)
user_df     = read_jsonl(user_path)

print("Done.\n")


# ---------------------------
# Basic dataset size stats
# ---------------------------

print("========== BASIC COUNTS ==========")
print(f"Businesses: {len(business_df):,}")
print(f"Users:      {len(user_df):,}")
print(f"Reviews:    {len(review_df):,}")
print(f"Tips:       {len(tip_df):,}")
print(f"Checkins:   {len(checkin_df):,}")

print("\nUnique IDs in interactions:")
print(f"  Review users:     {review_df['user_id'].nunique():,}")
print(f"  Review businesses:{review_df['business_id'].nunique():,}")
print(f"  Tip users:        {tip_df['user_id'].nunique():,}")
print(f"  Tip businesses:   {tip_df['business_id'].nunique():,}")


# ---------------------------
# Time frame stats
# ---------------------------

print("\n========== TIME FRAME ==========")

review_df["date"] = pd.to_datetime(review_df["date"], errors="coerce")
tip_df["date"] = pd.to_datetime(tip_df["date"], errors="coerce")

print(f"Review timeframe: {review_df['date'].min()} → {review_df['date'].max()}")
print(f"Tip timeframe:    {tip_df['date'].min()} → {tip_df['date'].max()}")

# interactions over time
reviews_per_month = review_df.set_index("date").resample("M").size()
tips_per_month = tip_df.set_index("date").resample("M").size()

print(f"\nReviews per month: mean={reviews_per_month.mean():.1f}, "
      f"median={reviews_per_month.median():.1f}, "
      f"max={reviews_per_month.max():.1f}")

print(f"Tips per month:    mean={tips_per_month.mean():.1f}, "
      f"median={tips_per_month.median():.1f}, "
      f"max={tips_per_month.max():.1f}")

# user tenure based on reviews
user_first = review_df.groupby("user_id")["date"].min()
user_last  = review_df.groupby("user_id")["date"].max()
user_tenure_days = (user_last - user_first).dt.days

describe_distribution(user_tenure_days, "User tenure (days)")


# ---------------------------
# Geography stats
# ---------------------------

print("\n========== GEOGRAPHY ==========")
if {"city", "state"}.issubset(business_df.columns):
    n_cities = business_df["city"].nunique()
    n_states = business_df["state"].nunique()
    print(f"Cities: {n_cities:,}")
    print(f"States/regions: {n_states:,}")

    biz_per_city = business_df["city"].value_counts()
    print("\nTop 10 cities by # businesses:")
    print(biz_per_city.head(10).to_string())

    # reviews per city (join review->business)
    review_city = review_df.merge(
        business_df[["business_id", "city"]], on="business_id", how="left"
    )
    reviews_per_city = review_city["city"].value_counts()

    print("\nTop 10 cities by # reviews:")
    print(reviews_per_city.head(10).to_string())

    top5_share = reviews_per_city.head(5).sum() / len(review_df)
    print(f"\nShare of all reviews in top 5 cities: {top5_share*100:.2f}%")
    print(f"City-review Gini (skewness): {gini(reviews_per_city.values):.3f}")
else:
    print("City/state not found in business file.")


# ---------------------------
# Sparsity / long tail stats
# ---------------------------

print("\n========== SPARSITY / LONG TAIL ==========")

reviews_per_user = review_df["user_id"].value_counts()
reviews_per_business = review_df["business_id"].value_counts()

describe_distribution(reviews_per_user, "Reviews per user")
describe_distribution(reviews_per_business, "Reviews per business")

density = len(review_df) / (review_df["user_id"].nunique() * review_df["business_id"].nunique())
print(f"\nUser–business matrix density: {density:.8f}")

cold_users = (reviews_per_user < 2).mean() * 100
cold_items = (reviews_per_business < 2).mean() * 100

print(f"Cold-start users (<2 reviews): {cold_users:.2f}%")
print(f"Cold-start businesses (<2 reviews): {cold_items:.2f}%")

head_share = reviews_per_business.head(int(0.1 * len(reviews_per_business))).sum() / len(review_df)
print(f"Top 10% businesses contain {head_share*100:.2f}% of all reviews")


# ---------------------------
# Sequential / session suitability
# ---------------------------

print("\n========== SEQUENTIAL SUITABILITY ==========")

# sequence length per user (chronological)
seq_len = review_df.sort_values("date").groupby("user_id").size()
describe_distribution(seq_len, "Sequence length per user (#reviews)")

print("\nSequential usability heuristics:")
print(f"  % users with >=5 reviews:  {(seq_len >= 5).mean()*100:.2f}%")
print(f"  % users with >=10 reviews: {(seq_len >= 10).mean()*100:.2f}%")
print(f"  % users with >=20 reviews: {(seq_len >= 20).mean()*100:.2f}%")


print("\n========== SESSION SUITABILITY ==========")

# simple sessionization: new session if gap > 1 day
review_df = review_df.sort_values(["user_id", "date"])
review_df["prev_date"] = review_df.groupby("user_id")["date"].shift(1)
review_df["gap_days"] = (review_df["date"] - review_df["prev_date"]).dt.days
review_df["new_session"] = (review_df["gap_days"].isna()) | (review_df["gap_days"] > 1)
review_df["session_id"] = review_df.groupby("user_id")["new_session"].cumsum()

sessions = review_df.groupby(["user_id", "session_id"]).size()

describe_distribution(sessions, "Session length (#reviews within session)")

print("\nSession usability heuristics:")
print(f"  Total sessions: {len(sessions):,}")
print(f"  Avg sessions per user: {sessions.groupby('user_id').size().mean():.2f}")
print(f"  % sessions length 1: {(sessions == 1).mean()*100:.2f}%")
print(f"  % sessions length >=2: {(sessions >= 2).mean()*100:.2f}%")


# ---------------------------
# Category richness (content side-info)
# ---------------------------

print("\n========== CATEGORY RICHNESS ==========")

if "categories" in business_df.columns:
    business_df["categories_list"] = business_df["categories"].fillna("").apply(
        lambda x: [c.strip() for c in x.split(",") if c.strip()]
    )
    cat_counts = business_df["categories_list"].explode().value_counts()

    print(f"Unique categories: {cat_counts.shape[0]:,}")
    print(f"Avg categories per business: {business_df['categories_list'].apply(len).mean():.2f}")

    print("\nTop 15 categories:")
    print(cat_counts.head(15).to_string())
else:
    print("Categories column not found.")

print("\n========== DONE ==========")
