#!/usr/bin/env python3
"""
Dataset statistics for IJCAI-15 Repeat Buyers (format1)

Expected files in the same folder:
- user_info_format1.csv
- user_log_format1.csv

Run:
  python stats_ijcai15.py
or:
  python stats_ijcai15.py --data-dir /path/to/folder
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def basic_frame_report(df: pd.DataFrame, name: str) -> None:
    print_header(f"{name}: basic info")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Missing values
    na = df.isna().sum().sort_values(ascending=False)
    na = na[na > 0]
    if len(na) == 0:
        print("Missing values: none")
    else:
        print("\nMissing values (count, %):")
        pct = (na / len(df) * 100).round(2)
        print(pd.DataFrame({"missing": na, "pct": pct}).to_string())

    # Duplicates
    dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup:,} ({dup/len(df)*100:.2f}%)")

    # Unique counts
    print("\nUnique values per column:")
    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    print(nunique.to_string())

    # Dtypes
    print("\nDtypes:")
    print(df.dtypes.to_string())


def top_k_counts(df: pd.DataFrame, col: str, k: int = 10) -> None:
    if col not in df.columns:
        return
    print_header(f"Top {k} most frequent: {col}")
    vc = df[col].value_counts(dropna=False).head(k)
    print(vc.to_string())


def numeric_describe(df: pd.DataFrame, name: str) -> None:
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        return
    print_header(f"{name}: numeric describe()")
    print(num.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T.to_string())


def safe_read_csv(path: Path) -> pd.DataFrame:
    # Common Tianchi CSVs are comma-separated, sometimes large.
    # low_memory=False gives better dtype inference.
    return pd.read_csv(path, low_memory=False)


def parse_time_stamp(series: pd.Series) -> pd.Series:
    """
    time_stamp in this dataset is usually mmdd (e.g., 1111 for Nov 11).
    We'll parse it to a datetime in a dummy year (2015) for range checks.
    """
    s = series.astype("string").str.strip()
    # Ensure 4 digits (mmdd)
    s = s.str.zfill(4)

    # Convert mmdd -> 2015-mm-dd
    mm = s.str.slice(0, 2)
    dd = s.str.slice(2, 4)
    dt = pd.to_datetime("2015-" + mm + "-" + dd, errors="coerce")
    return dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".", help="Folder containing the CSV files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    user_info_path = data_dir / "user_info_format1.csv"
    user_log_path = data_dir / "user_log_format1.csv"

    if not user_info_path.exists():
        raise FileNotFoundError(f"Missing: {user_info_path}")
    if not user_log_path.exists():
        raise FileNotFoundError(f"Missing: {user_log_path}")

    # Read
    print_header("Reading CSV files")
    user_info = safe_read_csv(user_info_path)
    user_log = safe_read_csv(user_log_path)
    print(f"Loaded {user_info_path.name}: {user_info.shape[0]:,} rows")
    print(f"Loaded {user_log_path.name}: {user_log.shape[0]:,} rows")

    # Basic reports
    basic_frame_report(user_info, "user_info_format1.csv")
    numeric_describe(user_info, "user_info_format1.csv")

    basic_frame_report(user_log, "user_log_format1.csv")
    numeric_describe(user_log, "user_log_format1.csv")

    # Expected columns (but keep it robust if names differ)
    # user_info: user_id, age_range, gender
    for c in ["age_range", "gender"]:
        top_k_counts(user_info, c, k=15)

    # user_log: user_id, item_id, cat_id, merchant_id, brand_id, time_stamp, action_type
    for c in ["action_type", "time_stamp"]:
        top_k_counts(user_log, c, k=15)

    # Time range (if time_stamp exists)
    if "time_stamp" in user_log.columns:
        print_header("Time stamp range")
        dt = parse_time_stamp(user_log["time_stamp"])
        valid = dt.dropna()
        if len(valid) == 0:
            print("Could not parse any time_stamp values.")
        else:
            print(f"Parsed timestamps: {len(valid):,}/{len(user_log):,}")
            print(f"Min date: {valid.min().date()}")
            print(f"Max date: {valid.max().date()}")

    # Activity volume stats
    if "user_id" in user_log.columns:
        print_header("Activity per user (distribution)")
        per_user = user_log.groupby("user_id").size()
        print(per_user.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
        print("\nTop 10 users by activity:")
        print(per_user.sort_values(ascending=False).head(10).to_string())

    if "merchant_id" in user_log.columns:
        print_header("Activity per merchant (distribution)")
        per_merchant = user_log.groupby("merchant_id").size()
        print(per_merchant.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
        print("\nTop 10 merchants by activity:")
        print(per_merchant.sort_values(ascending=False).head(10).to_string())

    # Per-action_type breakdown
    if "action_type" in user_log.columns:
        print_header("Action type breakdown")
        action_counts = user_log["action_type"].value_counts(dropna=False)
        print(action_counts.to_string())

        # Action type per user (how many different actions each user performed)
        if "user_id" in user_log.columns:
            print_header("Distinct action types per user")
            distinct_actions = user_log.groupby("user_id")["action_type"].nunique(dropna=True)
            print(distinct_actions.describe().to_string())

    # Distinct entity counts
    print_header("Distinct entity counts (if columns exist)")
    for col in ["user_id", "merchant_id", "item_id", "cat_id", "brand_id"]:
        if col in user_log.columns:
            print(f"{col}: {user_log[col].nunique(dropna=True):,}")

    # Quick join sanity check (how many log users have profiles)
    if "user_id" in user_log.columns and "user_id" in user_info.columns:
        print_header("Join coverage: log users with profiles")
        log_users = set(user_log["user_id"].dropna().unique())
        info_users = set(user_info["user_id"].dropna().unique())
        covered = len(log_users & info_users)
        print(f"Unique users in logs: {len(log_users):,}")
        print(f"Unique users in user_info: {len(info_users):,}")
        print(f"Log users with a profile row: {covered:,} ({covered/len(log_users)*100:.2f}%)")

    print_header("Done")


if __name__ == "__main__":
    main()
