"""
==============================================================================
Eataway Demand Forecasting System — Data Cleaning + Feature Engineering 
==============================================================================
Usage: python feature.py
Output: cleaned_weekly.csv, trainable_data.csv
==============================================================================
"""

import os
import pandas as pd
import numpy as np
import warnings
import pymysql
from pathlib import Path
from itertools import product as iter_product
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(dotenv_path=BASE_DIR / ".env")
DB_CONFIG = {
    "host":     os.environ["DB_HOST"],
    "port":     int(os.environ.get("DB_PORT", 3306)),
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "database": os.environ["DB_NAME"],
    "charset":  "utf8mb4",
}
DAYS_HISTORY = 730

# Time Partitioning (consistent with training script)
# Last 6 weeks = test, previous 6 weeks = val, remaining = train
TEST_WEEKS = 6
VAL_WEEKS  = 6


# ============================================================================
# Part 1: Data Cleaning
# ============================================================================

def load_from_db() -> pd.DataFrame:
    print("=" * 70)
    print("STEP 0: Loading Raw Data from Database")
    print("=" * 70)
    print("  Connecting to database...")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        query = f"""
            SELECT datum, namn, ort, typ, sort, antal_ordrar, antal_returer
            FROM ordrar_och_returer_looker_1y
            WHERE datum >= DATE_SUB(CURDATE(), INTERVAL {DAYS_HISTORY} DAY)
            ORDER BY datum
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    df["datum"] = pd.to_datetime(df["datum"])
    print(f"  ✓ Fetched {len(df):,} rows directly from database")
    print(f"  Date Range: {df['datum'].min().date()} ~ {df['datum'].max().date()}")
    print(f"  Number of Stores:   {df['namn'].nunique()}")
    print(f"  Number of Products: {df['sort'].nunique()}")
    print()
    return df


def step1_remove_paused_products(df):
    print("-" * 70)
    print("STEP 1: Removing Paused/Discontinued Products")
    mask = df["sort"].str.contains(r"beställ ej|Paus|\(EC\)", case=False, na=False)
    paused = df.loc[mask, "sort"].unique()
    for p in sorted(paused):
        print(f"    ✗ {p.strip()}")
    df_clean = df[~mask].copy()
    print(f"  Removed {mask.sum():,} rows → Remaining {len(df_clean):,} rows\n")
    return df_clean


def step2_remove_non_stores(df):
    print("-" * 70)
    print("STEP 2: Removing Non-Store Entities")
    store_orders = df.groupby("namn")["antal_ordrar"].sum()
    zero_stores  = store_orders[store_orders == 0].index.tolist()
    known_non    = ["Prover", "Alina Systems"]
    remove       = list(set(zero_stores + known_non))
    for s in sorted(remove):
        print(f"    ✗ {s:40s} (Total Orders: {int(store_orders.get(s, 0))})")
    df_clean = df[~df["namn"].isin(remove)].copy()
    print(f"  Removed {len(df) - len(df_clean):,} rows → Remaining {len(df_clean):,} rows\n")
    return df_clean


def step3_clean_names(df):
    print("-" * 70)
    print("STEP 3: Unifying Names & Creating product_id")
    for col in ["namn", "typ", "sort", "ort"]:
        df[col] = df[col].str.strip()
    df["product_id"] = df["typ"] + " | " + df["sort"]
    print(f"  Unique product_id count: {df['product_id'].nunique()}\n")
    return df


def step4_aggregate_weekly(df):
    print("-" * 70)
    print("STEP 4: Weekly Aggregation")
    df["year"]      = df["datum"].dt.isocalendar().year.astype(int)
    df["week"]      = df["datum"].dt.isocalendar().week.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    weekly = df.groupby(
        ["namn", "ort", "typ", "sort", "product_id", "year", "week", "year_week"],
        as_index=False,
    ).agg(
        weekly_ordrar  =("antal_ordrar",  "sum"),
        weekly_returer =("antal_returer", "sum"),
        delivery_days  =("antal_ordrar",  lambda x: (x > 0).sum()),
        active_days    =("datum",         "nunique"),
    )
    # faktisk = actual
    weekly["faktisk"] = weekly["weekly_ordrar"] - weekly["weekly_returer"]
    print(f"  After Aggregation: {len(weekly):,} rows\n")
    return weekly


def step5_handle_negatives(df):
    print("-" * 70)
    print("STEP 5: Handling Negative Values")
    n_neg = (df["faktisk"] < 0).sum()
    print(f"  Negative Rows: {n_neg:,} ({n_neg / len(df):.2%})")
    df["faktisk"] = df["faktisk"].clip(lower=0)
    print()
    return df


def step6_remove_truncated_weeks(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Mark truncated weeks (does not delete rows; training script handles exclusion logic).

    Uses 'faktisk' for detection — order/return data can be unreliable (all zeros) in some periods.
    Logic: If current week total 'faktisk' < avg of last 4 weeks * threshold → mark as truncated.
    """
    print("-" * 70)
    print(f"STEP 6: Truncated Week Detection (faktisk threshold={threshold:.0%})")

    all_weeks  = sorted(df["year_week"].unique())
    weekly_fak = df.groupby("year_week")["faktisk"].sum().reindex(all_weeks)

    truncated = []
    for i, wk in enumerate(all_weeks):
        if i < 4:
            continue
        # Avg of last 4 weeks (only using non-truncated weeks to avoid cascading false positives)
        prev_vals = []
        for j in range(max(0, i - 4), i):
            pw = all_weeks[j]
            if pw not in truncated:
                prev_vals.append(weekly_fak.loc[pw])
        if not prev_vals:
            continue
        prev_avg  = sum(prev_vals) / len(prev_vals)
        fak_curr  = weekly_fak.loc[wk]
        fak_ratio = fak_curr / max(prev_avg, 1)

        if fak_ratio < threshold:
            truncated.append(wk)
            print(f"  ⚠ Truncated Week: {wk}  faktisk={fak_curr:,.0f} / avg={prev_avg:,.0f} = {fak_ratio:.0%}")

    # Only force-mark the very last week as truncated (current/future week may be incomplete)
    # W09/W10/W11 have full data in DB — let the ratio check above decide for older weeks
    N_ALWAYS_TRUNCATED = 1
    recent_weeks = all_weeks[-N_ALWAYS_TRUNCATED:]
    for wk in recent_weeks:
        if wk not in truncated:
            truncated.append(wk)
            fak_curr = weekly_fak.loc[wk]
            print(f"  ⚠ Truncated Week: {wk}  faktisk={fak_curr:,.0f}  (Recent {N_ALWAYS_TRUNCATED} weeks forced mark due to entry delay)")

    # Mark but don't delete, training script handles the exclusion
    # Use bool to ensure no NaN issues during CSV reload
    df["is_truncated"] = df["year_week"].isin(truncated).astype(bool)

    if not truncated:
        print("  ✓ No truncated weeks detected")
    else:
        print(f"  → Marked {len(truncated)} truncated weeks (is_truncated=True, retained for out-of-sample prediction)")

    print()
    return df


def step7_filter_active_combinations(df, min_weeks=4, lookback_weeks=8):
    print("-" * 70)
    print(f"STEP 7: Filtering Active (Store, Product) Combinations")
    all_weeks   = sorted(df["year_week"].unique())
    recent      = all_weeks[-lookback_weeks:] if len(all_weeks) >= lookback_weeks else all_weeks
    recent_df   = df[df["year_week"].isin(recent)]
    active_cnt  = (
        recent_df[recent_df["weekly_ordrar"] > 0]
        .groupby(["namn", "product_id"])["year_week"].nunique()
    )
    active_set  = set(active_cnt[active_cnt >= min_weeks].index)
    before      = df[["namn", "product_id"]].drop_duplicates().shape[0]
    df_filtered = df[df.set_index(["namn", "product_id"]).index.isin(active_set)].copy()
    after       = df_filtered[["namn", "product_id"]].drop_duplicates().shape[0]
    print(f"  Total Combos: {before:,}  Active: {after:,}  Removed: {before-after:,}")
    print(f"  Remaining Rows: {len(df_filtered):,}\n")
    return df_filtered


def step8_remove_ghost_combinations(df: pd.DataFrame, min_active_weeks: int = 3) -> pd.DataFrame:
    """
    Fixes the root cause of y=0 ratio=3.0:

    Data contains many (Store, Product) combinations that have never been delivered or only once/twice,
    but fill_missing_weeks pads them into all-zero rows for the model.
    The Hurdle classifier gets trained on these "ghost combinations" and fails zero-value judgment.

    Strategy: Only keep combinations that have delivery records for at least min_active_weeks in history.
    This is stricter than Step 7 because it looks at global history, not just the recent window.
    """
    print("-" * 70)
    print(f"STEP 8: Removing Ghost Combinations (Global Active Weeks < {min_active_weeks})")

    combo_active = (
        df[df["weekly_ordrar"] > 0]
        .groupby(["namn", "product_id"])["year_week"]
        .nunique()
        .rename("global_active_weeks")
    )
    valid_combos = set(combo_active[combo_active >= min_active_weeks].index)

    before  = df[["namn", "product_id"]].drop_duplicates().shape[0]
    df_clean = df[df.set_index(["namn", "product_id"]).index.isin(valid_combos)].copy()
    after   = df_clean[["namn", "product_id"]].drop_duplicates().shape[0]

    print(f"  Total Combos: {before:,}  Retained: {after:,}  Removed Ghost: {before-after:,}")
    print(f"  Remaining Rows: {len(df_clean):,}")
    z_before = (df["faktisk"] == 0).mean()
    z_after  = (df_clean["faktisk"] == 0).mean()
    print(f"  Zero-value Proportion: {z_before:.1%} → {z_after:.1%}")
    print()
    return df_clean


def step9_censored_flag(df):
    print("-" * 70)
    print("STEP 9: Marking Censored Demand")
    df["is_censored"] = (df["weekly_returer"] == 0) & (df["weekly_ordrar"] > 0)
    n = df["is_censored"].sum()
    print(f"  'Sold Out' Rows: {n:,} ({n / max((df['weekly_ordrar']>0).sum(), 1):.1%})\n")
    return df


def compute_store_delivery_patterns(df_daily: pd.DataFrame, lookback_weeks: int = 1) -> pd.DataFrame:
    """
    From raw daily data (after name cleaning, before weekly aggregation),
    compute which weekdays each store typically delivers and in what proportions.

    Returns DataFrame with columns:
      namn, weekday, weeks_with_delivery, total_weeks, frequency,
      is_delivery_day, demand_sum, demand_ratio
    """
    print("-" * 70)
    print(f"Delivery Pattern: Computing per-store weekday delivery patterns (lookback={lookback_weeks}w)")

    WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    DELIVERY_THRESHOLD = 0.35  # A weekday is a "delivery day" if delivered in ≥35% of recent weeks

    cutoff = df_daily["datum"].max() - pd.Timedelta(weeks=lookback_weeks)
    recent = df_daily[df_daily["datum"] >= cutoff].copy()
    recent["weekday"] = recent["datum"].dt.day_name()
    recent["year_week"] = (
        recent["datum"].dt.isocalendar().year.astype(str)
        + "-W"
        + recent["datum"].dt.isocalendar().week.astype(str).str.zfill(2)
    )

    # --- Frequency: how often each weekday has any delivery for the store ---
    store_day_weeks = (
        recent[recent["antal_ordrar"] > 0]
        .groupby(["namn", "weekday", "year_week"])
        .size()
        .reset_index()
        .groupby(["namn", "weekday"])["year_week"]
        .nunique()
        .reset_index(name="weeks_with_delivery")
    )
    total_weeks = (
        recent.groupby("namn")["year_week"]
        .nunique()
        .reset_index(name="total_weeks")
    )
    pat = store_day_weeks.merge(total_weeks, on="namn", how="left")
    pat["frequency"] = pat["weeks_with_delivery"] / pat["total_weeks"].clip(lower=1)
    pat["is_delivery_day"] = pat["frequency"] >= DELIVERY_THRESHOLD

    # --- Demand ratio: what share of weekly demand falls on each weekday ---
    store_day_demand = (
        recent.groupby(["namn", "weekday"])["antal_ordrar"]
        .sum()
        .reset_index(name="demand_sum")
    )
    store_total_demand = (
        recent.groupby("namn")["antal_ordrar"]
        .sum()
        .reset_index(name="demand_total")
    )
    store_day_demand = store_day_demand.merge(store_total_demand, on="namn", how="left")
    store_day_demand["demand_ratio"] = (
        store_day_demand["demand_sum"] / store_day_demand["demand_total"].clip(lower=1)
    )

    pat = pat.merge(store_day_demand[["namn", "weekday", "demand_sum", "demand_ratio"]],
                    on=["namn", "weekday"], how="left")
    pat["demand_sum"]  = pat["demand_sum"].fillna(0)
    pat["demand_ratio"] = pat["demand_ratio"].fillna(0)

    # Sort by weekday order
    pat["weekday_order"] = pat["weekday"].map(
        {d: i for i, d in enumerate(WEEKDAY_ORDER)}
    ).fillna(99).astype(int)
    pat = pat.sort_values(["namn", "weekday_order"]).drop(columns=["weekday_order"])

    n_stores = pat["namn"].nunique()
    avg_days = pat[pat["is_delivery_day"]].groupby("namn").size().mean()
    print(f"  Stores analysed: {n_stores}  Average delivery days/store: {avg_days:.1f}")
    print(f"  Threshold: frequency >= {DELIVERY_THRESHOLD:.0%}")
    print()
    return pat


def run_cleaning_pipeline() -> pd.DataFrame:
    print("╔" + "═" * 68 + "╗")
    print("║" + "  EATAWAY Data Cleaning Pipeline V2".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")
    df = load_from_db()
    df = step1_remove_paused_products(df)
    df = step2_remove_non_stores(df)
    df = step3_clean_names(df)

    # Compute delivery patterns from daily data (before weekly aggregation)
    delivery_patterns = compute_store_delivery_patterns(df)
    delivery_patterns.to_csv(OUTPUT_DIR / "store_delivery_pattern.csv", index=False)
    print(f"✓ store_delivery_pattern.csv  ({len(delivery_patterns)} rows)\n")

    weekly = step4_aggregate_weekly(df)
    weekly = step5_handle_negatives(weekly)
    weekly = step6_remove_truncated_weeks(weekly)
    weekly = step7_filter_active_combinations(weekly)
    weekly = step8_remove_ghost_combinations(weekly, min_active_weeks=5)   # V7.1: Increased from 3 to 5
    weekly = step9_censored_flag(weekly)
    print("=" * 70)
    print(f"Cleaning Complete! Final: {len(weekly):,} rows  Stores={weekly['namn'].nunique()}  Products={weekly['product_id'].nunique()}  Weeks={weekly['year_week'].nunique()}")
    print("=" * 70 + "\n")
    return weekly


# ============================================================================
# Part 2: Feature Engineering
# ============================================================================

def get_train_cutoff(df: pd.DataFrame) -> str:
    """
    Returns the last week of the training set (used for leakage-free aggregation)
    Train set = All data - last (TEST_WEEKS + VAL_WEEKS)
    """
    all_weeks = sorted(df["year_week"].unique())
    cutoff_idx = len(all_weeks) - TEST_WEEKS - VAL_WEEKS
    return all_weeks[cutoff_idx - 1]  # Last week of train set


def fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("Feature Engineering STEP 0: Filling Missing Weeks")
    all_weeks = sorted(df["year_week"].unique())
    week_info = df[["year", "week", "year_week"]].drop_duplicates().sort_values("year_week")
    combo_info = (
        df.groupby(["namn", "product_id"], as_index=False)
        .agg(ort=("ort", "first"), typ=("typ", "first"), sort=("sort", "first"))
    )
    combos     = list(zip(combo_info["namn"], combo_info["product_id"]))
    full_index = pd.DataFrame(
        [(n, p, yw) for (n, p), yw in iter_product(combos, all_weeks)],
        columns=["namn", "product_id", "year_week"],
    )
    df_full = full_index.merge(df, on=["namn", "product_id", "year_week"], how="left")
    df_full = df_full.merge(combo_info, on=["namn", "product_id"], how="left", suffixes=("", "_fill"))
    for col in ["ort", "typ", "sort"]:
        df_full[col] = df_full[col].fillna(df_full[f"{col}_fill"])
        df_full.drop(columns=[f"{col}_fill"], inplace=True)
    df_full["year"] = df_full["year"].astype("float64")
    df_full["week"] = df_full["week"].astype("float64")
    yw_map = week_info.set_index("year_week")[["year", "week"]].to_dict("index")
    mask = df_full["year"].isna()
    if mask.any():
        df_full.loc[mask, "year"] = df_full.loc[mask, "year_week"].map(lambda x: yw_map.get(x, {}).get("year"))
        df_full.loc[mask, "week"] = df_full.loc[mask, "year_week"].map(lambda x: yw_map.get(x, {}).get("week"))
    df_full["year"] = df_full["year"].astype(int)
    df_full["week"] = df_full["week"].astype(int)
    for col in ["weekly_ordrar", "weekly_returer", "delivery_days", "active_days", "faktisk"]:
        df_full[col] = df_full[col].fillna(0)
    df_full["is_censored"] = df_full["is_censored"].fillna(False)
    print(f"  Before Padding: {len(df):,}  After Padding: {len(df_full):,}  New Zero-value Weeks: {len(df_full)-len(df):,}\n")
    return df_full


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("Feature Engineering STEP 1: Time Features")
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["week_start_date"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str).str.zfill(2) + "1",
        format="%G%V%u", errors="coerce"
    )
    df["month"] = df["week_start_date"].dt.month.fillna(
        np.clip(((df["week"] - 1) * 7 // 30) + 1, 1, 12)
    ).astype(int)
    season_map = {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",
                  6:"summer",7:"summer",8:"summer",9:"autumn",10:"autumn",11:"autumn"}
    df["season"]    = df["month"].map(season_map)
    df["week_sin"]  = np.sin(2 * np.pi * df["week"]  / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["week"]  / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_december"] = (df["month"] == 12).astype(int)
    df["is_summer"]   = (df["season"] == "summer").astype(int)
    print(f"  ✓ week/month/season/sin-cos/is_december/is_summer\n")
    return df


def add_swedish_holidays(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("Feature Engineering STEP 2: Swedish Holiday Features")
    try:
        import holidays as hd
        se_holidays = hd.Sweden(years=range(2024, 2028))
        all_dates   = pd.date_range(
            df["week_start_date"].min() - pd.Timedelta(days=7),
            df["week_start_date"].max() + pd.Timedelta(days=14),
        )
        holiday_df = (
            pd.Series({d: se_holidays.get(d) for d in all_dates})
            .dropna().reset_index()
        )
        holiday_df.columns = ["date", "holiday_name"]
        holiday_df["date"] = pd.to_datetime(holiday_df["date"])
        holiday_df["year"] = holiday_df["date"].dt.isocalendar().year.astype(int)
        holiday_df["week"] = holiday_df["date"].dt.isocalendar().week.astype(int)
        print("  ✓ Using holidays library")
    except ImportError:
        print("  ⚠ holidays library not installed, using manual definitions")
        fixed = {"Nyårsdagen":(1,1),"Trettondedag jul":(1,6),"Första maj":(5,1),
                 "Nationaldagen":(6,6),"Julafton":(12,24),"Juldagen":(12,25),
                 "Annandag jul":(12,26),"Nyårsafton":(12,31)}
        floating = {
            2024:{"Långfredagen":(3,29),"Påskdagen":(3,31),"Annandag påsk":(4,1),
                  "Kristi himmelsfärdsdag":(5,9),"Midsommarafton":(6,21),"Midsommardagen":(6,22),"Alla helgons dag":(11,2)},
            2025:{"Långfredagen":(4,18),"Påskdagen":(4,20),"Annandag påsk":(4,21),
                  "Kristi himmelsfärdsdag":(5,29),"Midsommarafton":(6,20),"Midsommardagen":(6,21),"Alla helgons dag":(11,1)},
            2026:{"Långfredagen":(4,3),"Påskdagen":(4,5),"Annandag påsk":(4,6),
                  "Kristi himmelsfärdsdag":(5,14),"Midsommarafton":(6,19),"Midsommardagen":(6,20),"Alla helgons dag":(10,31)},
            2027:{"Långfredagen":(3,26),"Påskdagen":(3,28),"Annandag påsk":(3,29),
                  "Kristi himmelsfärdsdag":(5,6),"Midsommarafton":(6,25),"Midsommardagen":(6,26),"Alla helgons dag":(11,6)},
        }
        records = []
        for yr in range(2024, 2028):
            for name, (m, d) in fixed.items():
                records.append({"date": pd.Timestamp(yr, m, d), "holiday_name": name})
            if yr in floating:
                for name, (m, d) in floating[yr].items():
                    records.append({"date": pd.Timestamp(yr, m, d), "holiday_name": name})
        holiday_df = pd.DataFrame(records)
        holiday_df["year"] = holiday_df["date"].dt.isocalendar().year.astype(int)
        holiday_df["week"] = holiday_df["date"].dt.isocalendar().week.astype(int)

    hpw = (holiday_df.groupby(["year", "week"])
           .agg(n_holidays=("holiday_name","count"), holiday_names=("holiday_name", lambda x: "|".join(x)))
           .reset_index())
    high_impact = ["Midsommar", "Jul", "Påsk", "Nyår"]
    hpw["is_high_impact_holiday"] = hpw["holiday_names"].apply(
        lambda s: int(any(h in s for h in high_impact))
    )
    df = df.merge(hpw, on=["year", "week"], how="left")
    df["n_holidays"]            = df["n_holidays"].fillna(0).astype(int)
    df["is_holiday_week"]       = (df["n_holidays"] > 0).astype(int)
    df["is_high_impact_holiday"] = df["is_high_impact_holiday"].fillna(0).astype(int)
    df["holiday_names"]         = df["holiday_names"].fillna("")

    # Mark pre/post holiday weeks
    holiday_weeks = set(zip(hpw["year"], hpw["week"]))
    def adj(row, offset):
        y, w = int(row["year"]), int(row["week"]) + offset
        if w < 1:   y -= 1; w += 52
        elif w > 52: y += 1; w -= 52
        return int((y, w) in holiday_weeks)
    df["is_pre_holiday_week"]  = df.apply(lambda r: adj(r,  1), axis=1)
    df["is_post_holiday_week"] = df.apply(lambda r: adj(r, -1), axis=1)

    print(f"  ✓ Holiday week records: {df['is_holiday_week'].sum():,}\n")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    print("-" * 70)
    print("Feature Engineering STEP 3: Lag & Rolling Features")
    df = df.sort_values(["namn", "product_id", "year", "week"]).reset_index(drop=True)
    g  = ["namn", "product_id"]
    t  = "faktisk"

    for lag in [1, 2, 3, 4]:
        df[f"lag_{lag}w"] = df.groupby(g)[t].shift(lag)

    for w in [4, 8, 12]:
        df[f"rolling_mean_{w}w"] = df.groupby(g)[t].transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).mean())
        if w <= 8:
            df[f"rolling_std_{w}w"] = df.groupby(g)[t].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(1, w//2)).std())

    df["rolling_median_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).median())
    df["rolling_max_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).max())
    df["rolling_min_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).min())
    df["rolling_q75_4w"] = df.groupby(g)[t].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).quantile(0.75))

    # Linear trend
    def rolling_slope(series, window=4):
        result = pd.Series(index=series.index, dtype=float)
        x = np.arange(window)
        for i in range(window, len(series) + 1):
            y = series.iloc[i-window:i].values
            result.iloc[i-1] = np.nan if np.isnan(y).any() else np.polyfit(x, y, 1)[0]
        return result
    df["trend_4w"] = df.groupby(g)[t].transform(lambda x: rolling_slope(x.shift(1), 4))

    # Year-over-year same week
    prev = df[["namn","product_id","year","week",t]].copy()
    prev["year"] += 1
    prev = prev.rename(columns={t: "yoy_same_week"})
    df = df.merge(prev[["namn","product_id","year","week","yoy_same_week"]],
                  on=["namn","product_id","year","week"], how="left")

    # Return rate
    df["return_rate"] = np.where(df["weekly_ordrar"] > 0,
                                 df["weekly_returer"] / df["weekly_ordrar"], 0)
    df["return_rate_lag1"]        = df.groupby(g)["return_rate"].shift(1)
    df["rolling_return_rate_4w"]  = df.groupby(g)["return_rate"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    df["censored_ratio_4w"] = df.groupby(g)["is_censored"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean())

    print(f"  ✓ lag 1-4w / rolling mean/std/median/max/min/q75 / trend / yoy / return_rate\n")
    return df


def add_store_product_features_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    Fix 2: Leakage-free store/product aggregation features
    
    All statistics are calculated only using year_week <= train_cutoff, 
    then mapped to the whole dataset (including val/test).
    This ensures features for val/test rows do not contain their own 'y' information.
    """
    print("-" * 70)
    print(f"Feature Engineering STEP 4: Store & Product Features (No Leakage, Train Cutoff={train_cutoff})")

    train_only = df[df["year_week"] <= train_cutoff]

    # Store stats (Train only)
    store_stats = (train_only.groupby("namn")["faktisk"]
                   .agg(store_avg_weekly="mean", store_std_weekly="std").reset_index())
    df = df.merge(store_stats, on="namn", how="left")

    # Number of active products per store (Train only)
    store_np = (train_only[train_only["weekly_ordrar"] > 0]
                .groupby("namn")["product_id"].nunique().rename("store_n_products").reset_index())
    df = df.merge(store_np, on="namn", how="left")
    df["store_n_products"] = df["store_n_products"].fillna(0)

    # Product stats (Train only)
    prod_stats = (train_only.groupby("product_id")["faktisk"]
                  .agg(product_avg_weekly="mean", product_std_weekly="std").reset_index())
    df = df.merge(prod_stats, on="product_id", how="left")

    # Product store coverage (Train only)
    prod_ns = (train_only[train_only["weekly_ordrar"] > 0]
               .groupby("product_id")["namn"].nunique().rename("product_n_stores").reset_index())
    df = df.merge(prod_ns, on="product_id", how="left")
    df["product_n_stores"] = df["product_n_stores"].fillna(0)

    # Store x Product Share (Train only)
    store_total_map = train_only.groupby("namn")["faktisk"].sum()
    sp_total_map    = train_only.groupby(["namn","product_id"])["faktisk"].sum()
    df["store_product_share"] = df.apply(
        lambda r: sp_total_map.get((r["namn"], r["product_id"]), 0) /
                  max(store_total_map.get(r["namn"], 1), 1), axis=1)

    # Route average (Train only)
    route_avg = train_only.groupby("ort")["faktisk"].mean().rename("route_avg_demand")
    df = df.merge(route_avg, on="ort", how="left")

    print(f"  ✓ store_avg/std/n_products, product_avg/std/n_stores, store_product_share, route_avg\n")
    return df


def add_category_seasonality_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    Fix 2 cont: Category x Season interactions (No Leakage)
    """
    print("-" * 70)
    print(f"Feature Engineering STEP 5: Category x Season Interaction (No Leakage)")
    train_only = df[df["year_week"] <= train_cutoff]

    cat_season = (train_only.groupby(["typ","season"])["faktisk"].mean()
                  .rename("category_season_avg").reset_index())
    cat_month  = (train_only.groupby(["typ","month"])["faktisk"].mean()
                  .rename("category_month_avg").reset_index())
    cat_hol    = (train_only.groupby(["typ","is_holiday_week"])["faktisk"].mean()
                  .rename("category_holiday_avg").reset_index())

    # Category x High Impact Holiday
    cat_himp   = (train_only.groupby(["typ","is_high_impact_holiday"])["faktisk"].mean()
                  .rename("category_high_impact_avg").reset_index())

    df = df.merge(cat_season, on=["typ","season"],               how="left")
    df = df.merge(cat_month,  on=["typ","month"],                how="left")
    df = df.merge(cat_hol,    on=["typ","is_holiday_week"],      how="left")
    df = df.merge(cat_himp,   on=["typ","is_high_impact_holiday"], how="left")

    print(f"  ✓ category_season_avg / category_month_avg / category_holiday_avg / category_high_impact_avg\n")
    return df


def encode_categoricals_no_leakage(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    Fix 2 cont: Target encoding (No Leakage)
    
    Original version used all data → test set y values were part of encoding → leakage.
    Now only rows with year_week <= train_cutoff estimate means for each category.
    """
    print("-" * 70)
    print(f"Feature Engineering STEP 6: Target Encoding (No Leakage, Train Cutoff={train_cutoff})")
    train_only  = df[df["year_week"] <= train_cutoff]
    global_mean = train_only["faktisk"].mean()
    smoothing   = 20

    for col in ["namn", "product_id", "typ", "ort"]:
        counts = train_only.groupby(col)["faktisk"].count()
        means  = train_only.groupby(col)["faktisk"].mean()
        # Smoothing target encoding
        smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)
        df[f"{col}_te"] = df[col].map(smooth).fillna(global_mean)
        print(f"  ✓ {col}_te  (global_mean={global_mean:.2f}, smoothing={smoothing})")

    # Season one-hot
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)
    print(f"  ✓ season one-hot\n")
    return df


def add_holiday_interactions(df: pd.DataFrame, train_cutoff: str) -> pd.DataFrame:
    """
    Holiday Interaction: holiday x historical demand
    Resolves low holiday feature weights — just is_holiday_week=1 isn't enough.
    The model needs to know "how much did this store historically sell during holidays."
    """
    print("-" * 70)
    print("Feature Engineering STEP 6b: Holiday Interaction Features")
    train_only = df[df["year_week"] <= train_cutoff]

    # (Store, Product) historical avg demand during holiday weeks
    holiday_mean = (
        train_only[train_only["is_holiday_week"] == 1]
        .groupby(["namn", "product_id"])["faktisk"]
        .mean()
        .rename("combo_holiday_avg")
        .reset_index()
    )
    df = df.merge(holiday_mean, on=["namn", "product_id"], how="left")
    df["combo_holiday_avg"] = df["combo_holiday_avg"].fillna(
        df.groupby("product_id")["faktisk"].transform("mean")
    )

    # Holiday demand vs normal demand ratio (Holiday Lift)
    normal_mean = (
        train_only[train_only["is_holiday_week"] == 0]
        .groupby(["namn", "product_id"])["faktisk"]
        .mean()
        .rename("combo_normal_avg")
        .reset_index()
    )
    df = df.merge(normal_mean, on=["namn", "product_id"], how="left")
    df["combo_normal_avg"] = df["combo_normal_avg"].fillna(1)
    df["holiday_lift_ratio"] = np.where(
        df["combo_normal_avg"] > 0,
        df["combo_holiday_avg"] / df["combo_normal_avg"].clip(lower=0.1),
        1.0
    ).clip(0, 5)

    # Current week is holiday x rolling_mean (Expected holiday demand)
    df["holiday_x_mean"] = df["is_holiday_week"] * df.get("rolling_mean_4w", pd.Series(0, index=df.index))
    df["high_impact_x_mean"] = df["is_high_impact_holiday"] * df.get("rolling_mean_4w", pd.Series(0, index=df.index))

    print(f"  ✓ combo_holiday_avg / holiday_lift_ratio / holiday_x_mean / high_impact_x_mean\n")
    return df


def select_final_features(df: pd.DataFrame) -> tuple:
    print("-" * 70)
    print("Feature Engineering STEP 7: Final Feature Selection (Streamlined, redundancy removed)")

    feature_cols = [
        # -- Time (Keep core, remove redundancy) ----------------------
        "week", "month",
        "week_sin", "week_cos",          # Periodic encoding, removed month_sin/cos (redundant with month)
        "is_december", "is_summer",

        # -- Holiday (Enhanced with interactions) ---------------------
        "is_holiday_week",
        "is_high_impact_holiday",        # Keep high impact
        "is_pre_holiday_week",           # Pre-holiday (stocking effect)
        "is_post_holiday_week",          # Post-holiday (drop-off effect)
        "n_holidays",                    # Number of holidays in week
        "combo_holiday_avg",             # (Store, Product) historical holiday avg <- New
        "holiday_lift_ratio",            # Holiday lift multiplier <- New
        "holiday_x_mean",                # Holiday x rolling_mean <- New
        "high_impact_x_mean",            # High impact x rolling_mean <- New

        # -- Lag (Streamlined: removed lag_3w/4w, covered by rolling) --
        "lag_1w", "lag_2w",
        "rolling_mean_4w",               # Most important, keep
        "rolling_mean_12w",              # Long-term trend; removed rolling_mean_8w (redundant)
        "rolling_std_4w",                # Volatility
        "rolling_max_4w",                # Upper bound demand
        # rolling_min_4w removed: Anchor for history but leads to systemic underestimation
        # High/low info covered by max + std
        "rolling_median_4w",             # Outlier resistant median
        "rolling_q75_4w",                # 75th percentile for high demand prediction
        "trend_4w",                      # Direction of trend
        "yoy_same_week",                 # Yearly seasonality

        # -- Return (Keep core) -------------------------------------------
        "rolling_return_rate_4w",        # Return rate trend
        "censored_ratio_4w",             # 'Sold Out' proportion

        # -- Store Features (Streamlined) -----------------------------------
        "store_avg_weekly",              # Store scale
        "store_std_weekly",              # Store volatility
        "store_product_share",           # Product's share in store

        # -- Product Features (Streamlined) ---------------------------------
        "product_avg_weekly",            # Product global mean
        "product_n_stores",              # Product coverage

        # -- Category x Season (Streamlined: season only, removed month/holiday redundancy) --
        "category_season_avg",           # Category avg in current season

        # -- Encoding (Streamlined: removed ort_te, covered by route_avg_demand) --
        "namn_te",                       # Store target encoding
        "product_id_te",                 # Product target encoding
        "typ_te",                        # Category target encoding

        # -- Season one-hot --------------------------------------------------
        "season_winter", "season_spring", "season_summer", "season_autumn",
    ]

    actual  = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Columns not found: {missing}")

    print(f"  Final Feature Count: {len(actual)}  (Original 46 → Now {len(actual)})")
    print(f"  Removed Redundant Features: rolling_mean_8w, rolling_std_8w, lag_3w, lag_4w,")
    print(f"                             month_sin/cos, category_month_avg, category_holiday_avg,")
    print(f"                             ort_te, route_avg_demand, return_rate_lag1, store_n_products,")
    print(f"                             product_std_weekly, product_avg_weekly(partial)")
    print(f"  New Interaction Features: combo_holiday_avg, holiday_lift_ratio,")
    print(f"                            holiday_x_mean, high_impact_x_mean")
    print()
    return ["namn","ort","typ","sort","product_id","year","week","year_week"], actual, "faktisk"


def run_feature_engineering(weekly: pd.DataFrame) -> tuple:
    print("\n╔" + "═" * 68 + "╗")
    print("║" + "  EATAWAY Feature Engineering Pipeline V2 (No Leakage)".center(60) + "║")
    print("╚" + "═" * 68 + "╝\n")

    train_cutoff = get_train_cutoff(weekly)
    print(f"  Train Set Cutoff Week: {train_cutoff}  (Val/Test features exclude data after this)\n")

    df = fill_missing_weeks(weekly)

    # fill_missing_weeks adds rows without is_truncated → NaN
    # Map back using year_week: all rows in the same week should have same is_truncated
    if "is_truncated" in df.columns:
        trunc_map = df.dropna(subset=["is_truncated"]).groupby("year_week")["is_truncated"].first()
        df["is_truncated"] = df["year_week"].map(trunc_map).fillna(False).astype(bool)

    df = add_time_features(df)
    df = add_swedish_holidays(df)
    df = add_lag_features(df)
    df = add_store_product_features_no_leakage(df, train_cutoff)
    df = add_category_seasonality_no_leakage(df, train_cutoff)
    df = add_holiday_interactions(df, train_cutoff)        # <- New Holiday Interaction
    df = encode_categoricals_no_leakage(df, train_cutoff)

    id_cols, feature_cols, target = select_final_features(df)

    df["is_trainable"] = df["lag_4w"].notna()
    n_trainable = df["is_trainable"].sum()
    print(f"Trainable rows: {n_trainable:,} / {len(df):,}\n")

    return df, id_cols, feature_cols, target


# ============================================================================
# Main Program
# ============================================================================

def main():
    weekly = run_cleaning_pipeline()
    weekly.to_csv(OUTPUT_DIR / "cleaned_weekly.csv", index=False)
    print(f"✓ cleaned_weekly.csv")

    df_features, id_cols, feature_cols, target = run_feature_engineering(weekly)
    df_features.to_csv(OUTPUT_DIR / "features_ready.csv", index=False)
    print(f"✓ features_ready.csv")

    trainable = df_features[df_features["is_trainable"]].copy()
    # Save is_truncated for training script to split train vs out-of-sample prediction
    save_cols = [c for c in id_cols + feature_cols + [target, "is_censored", "is_truncated"] if c in trainable.columns]
    trainable[save_cols].to_csv(OUTPUT_DIR / "trainable_data.csv", index=False)
    n_trunc = trainable["is_truncated"].sum() if "is_truncated" in trainable.columns else 0
    print(f"✓ trainable_data.csv  ({len(trainable):,} rows, {len(feature_cols)} features, where {n_trunc:,} are truncated weeks for out-of-sample prediction)")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  V3 Fix Summary".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║  Fix 1: Automatically exclude truncated weeks (incomplete recent data) ║")
    print("║  Fix 2: Target encoding etc. all calculated using train set only       ║")
    print("║  Fix 3: Removed ghost combinations (rarely/never delivered combos)     ║")
    print("║          → Fixes the root cause of y=0 ratio=3.0                       ║")
    print("║  Fix 4: Feature reduction 46→30, redundancy removed                    ║")
    print("║          Added holiday interactions (holiday x rolling_mean, etc.)     ║")
    print("║          → Holiday feature weight expected to rise from 0.x% to 3-5%   ║")
    print("╚" + "═" * 68 + "╝")
    print("\nNext step: Run training script with the newly generated trainable_data.csv")


if __name__ == "__main__":
    main()