"""
Eataway Prediction System — Web Interface (Production Version)

Local run:  python app.py
Cloud deploy:  gunicorn --chdir eataway_system app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1

Optional password protection: set environment variable APP_PASSWORD=your_password
  → Uploading data and triggering training requires a password; viewing data does not

Date selection functionality:
  - Select a past date → shows real shipment data from DB (FAKTISK)
  - Select a future/current week → shows model prediction results (PROGNOS)
"""

import subprocess
import threading
import re
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # Max upload 200MB

# ── Path configuration ────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.parent
OUTPUT_DIR       = BASE_DIR / "output_v7"
FEATURE_SCRIPT   = BASE_DIR / "feature.py"
TRAIN_SCRIPT     = BASE_DIR / "eataway_train_v7.py"
DATA_FILE        = BASE_DIR / "1year.csv"        # fallback only (local dev)
DATA_ZIP         = BASE_DIR / "1year.csv.zip"    # fallback only (local dev)

# ── Database config (Railway: set these as environment variables) ──────────
_DB_CFG = {
    "host":     os.environ.get("DB_HOST"),
    "port":     int(os.environ.get("DB_PORT", 3306)),
    "user":     os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
    "charset":  "utf8mb4",
} if os.environ.get("DB_HOST") else None

DB_CACHE_SECONDS = 4 * 3600   # re-fetch from DB at most once every 4 hours

# ── Google Sheets configuration ───────────────────────────────
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
GSHEET_ID        = "1pRX_Mjc1Y_Xt1LYSposrLK7mkDSjMxCbF-IYDzixzlQ"
GSHEET_TAB       = "Tabell"

def _ensure_data_file():
    """If 1year.csv does not exist but 1year.csv.zip does, auto-extract it."""
    if not DATA_FILE.exists() and DATA_ZIP.exists():
        import zipfile
        try:
            with zipfile.ZipFile(DATA_ZIP, "r") as zf:
                zf.extract("1year.csv", BASE_DIR)
            print(f"✓ Auto-extracted {DATA_ZIP.name} → 1year.csv")
        except Exception as e:
            print(f"✗ Extraction failed: {e}")

# ── Optional password ─────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD", "eataway")

def _password_ok() -> bool:
    if not APP_PASSWORD:
        return True
    pw = (request.headers.get("X-App-Password")
          or request.args.get("pw")
          or (request.get_json(silent=True) or {}).get("pw")
          or request.form.get("pw"))
    return pw == APP_PASSWORD

# ── Pipeline status ───────────────────────────────────────────
pipeline_status = {"running": False, "log": [], "done": False, "error": False}


# ── Date utilities ────────────────────────────────────────────
MONTHS_SV   = ["jan","feb","mar","apr","maj","jun",
                "jul","aug","sep","okt","nov","dec"]
DAY_OFFSETS = {"Sunday":-1,"Monday":0,"Tuesday":1,"Wednesday":2,
               "Thursday":3,"Friday":4,"Saturday":5}
DAY_SHORT   = {"Sunday":"Sön","Monday":"Mån","Tuesday":"Tis","Wednesday":"Ons",
               "Thursday":"Tor","Friday":"Fre","Saturday":"Lör"}
DAY_ORDER   = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,
               "Thursday":4,"Friday":5,"Saturday":6}

def week_monday(year_week: str):
    m = re.match(r"(\d{4})-W(\d{2})", str(year_week))
    if not m:
        return None
    y, w = int(m.group(1)), int(m.group(2))
    return datetime.strptime(f"{y}-W{w:02d}-1", "%G-W%V-%u")

def fmt_d(d: datetime) -> str:
    return f"{d.day} {MONTHS_SV[d.month-1]}"

def week_to_range(yw: str) -> str:
    mon = week_monday(yw)
    if not mon:
        return yw
    sun = mon - timedelta(days=1)
    thu = mon + timedelta(days=3)
    return f"{fmt_d(sun)} – {fmt_d(thu)} {mon.year}"

def fmt_date_range(start_str: str, end_str: str) -> str:
    """Format a YYYY-MM-DD range as '9 mar – 15 mar 2026'."""
    try:
        s = datetime.strptime(start_str, "%Y-%m-%d")
        e = datetime.strptime(end_str,   "%Y-%m-%d")
        if start_str == end_str:
            return f"{s.day} {MONTHS_SV[s.month-1]} {s.year}"
        if s.year == e.year:
            return f"{s.day} {MONTHS_SV[s.month-1]} – {e.day} {MONTHS_SV[e.month-1]} {s.year}"
        return f"{s.day} {MONTHS_SV[s.month-1]} {s.year} – {e.day} {MONTHS_SV[e.month-1]} {e.year}"
    except Exception:
        return f"{start_str} – {end_str}"

def day_label(yw: str, day_name: str) -> str:
    mon = week_monday(yw)
    if not mon:
        return day_name
    d = mon + timedelta(days=DAY_OFFSETS.get(day_name, 0))
    return f"{DAY_SHORT.get(day_name, day_name)} {d.day}/{d.month}"

def week_step(yw: str, delta: int) -> str:
    """Advance a year_week string by delta weeks."""
    m = re.match(r"(\d{4})-W(\d{2})", yw)
    if not m:
        return yw
    mon = week_monday(yw)
    if not mon:
        return yw
    target = mon + timedelta(weeks=delta)
    iso = target.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ── Pipeline ──────────────────────────────────────────────────
def run_pipeline_thread():
    global pipeline_status
    pipeline_status = {"running": True, "log": [], "done": False, "error": False}
    python = sys.executable
    steps = [
        ("Feature Engineering", [python, str(FEATURE_SCRIPT)]),
        ("Model Training", [python, str(TRAIN_SCRIPT)]),
    ]
    for step_name, cmd in steps:
        pipeline_status["log"].append(f"\n▶ {step_name}...")
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(BASE_DIR),
                env=env,
            )
            for line in proc.stdout:
                pipeline_status["log"].append(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                pipeline_status["log"].append(f"✗ {step_name} failed (code={proc.returncode})")
                pipeline_status["error"] = True
                break
            else:
                pipeline_status["log"].append(f"✓ {step_name} complete")
        except Exception as e:
            pipeline_status["log"].append(f"✗ Error: {e}")
            pipeline_status["error"] = True
            break
    # ── Auto-export to Google Sheet after pipeline completes ──
    if not pipeline_status["error"]:
        pipeline_status["log"].append("\n▶ Exporting to Google Sheet...")
        gs_result = export_to_gsheet()
        icon = "✓" if gs_result["ok"] else "✗"
        pipeline_status["log"].append(f"{icon} {gs_result['msg']}")

    pipeline_status["running"] = False
    pipeline_status["done"]    = True


# ── Model prediction data (from output_v7/ CSVs) ─────────────
def load_csv(path):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None

def get_predicted_week() -> str:
    """Return the model's predicted week (future week).
    Prefer extracting from predictions/ filenames (most reliable);
    year_week in evaluation_v7.csv is historical evaluation data and cannot be used as the predicted week.
    """
    # ① Prefer extracting from predictions/ folder filenames (e.g. 2026-W11_driver.csv)
    pred_dir = BASE_DIR / "predictions"
    if pred_dir.exists():
        weeks = []
        for f in pred_dir.glob("*_driver.csv"):
            m = re.match(r"(\d{4}-W\d{2})_driver\.csv", f.name)
            if m:
                weeks.append(m.group(1))
        if weeks:
            return max(weeks)
    # ② Read from year_week column in driver_view_v7.csv (evaluation_v7 unusable, contains historical data)
    df = load_csv(OUTPUT_DIR / "driver_view_v7.csv")
    if df is not None and "year_week" in df.columns:
        val = str(df["year_week"].max())
        if val not in ("nan", "", "None"):
            return val
    return ""

def get_driver_predicted(display_yw: str = "") -> dict:
    """Get driver view from model output CSV. display_yw overrides date labels."""
    df = load_csv(OUTPUT_DIR / "driver_view_v7.csv")
    if df is None:
        return {}
    yw = display_yw or (str(df["year_week"].max()) if "year_week" in df.columns else "")
    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    df = df.sort_values(["Route", "_ord", "Store"])
    result = {}
    for route, rdf in df.groupby("Route"):
        result[route] = {}
        for store, sdf in rdf.groupby("Store"):
            result[route][store] = []
            for _, row in sdf.sort_values("_ord").iterrows():
                result[route][store].append({
                    "day":      row["Day"],
                    "day_label": day_label(yw, row["Day"]),
                    "qty":      int(row["Total_Qty"]),
                    "range":    row["Range"],
                    "products": str(row.get("Products", "")),
                    "n":        int(row.get("N_Products", 0)),
                })
    return result

def get_kitchen_predicted(display_yw: str = "") -> dict:
    """Get kitchen view from model output CSV."""
    df = load_csv(OUTPUT_DIR / "kitchen_view_v7.csv")
    if df is None:
        return {}
    yw = display_yw or get_predicted_week()
    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    df = df.sort_values(["_ord", "Qty"], ascending=[True, False])
    result = {}
    for day, ddf in df.groupby("Day"):
        result[day] = {
            "label": day_label(yw, day),
            "rows": [
                {"type": r["Type"], "product": r["Product"],
                 "qty": int(r["Qty"]), "range": r["Range"], "stores": int(r["Stores"])}
                for _, r in ddf.iterrows()
            ]
        }
    return result

def get_metrics():
    df = load_csv(OUTPUT_DIR / "metrics_v7.csv")
    if df is None:
        return None
    return df.iloc[0].to_dict()


# ── Historical actual data ─────────────────────────────────────
_raw_state   = {"df": None, "mtime": None, "db_fetched_at": None}
_hist_cache  = {}   # year_week → {"driver": {…}, "kitchen": {…}}
_range_cache = {}   # (start, end) → {"driver": {…}, "kitchen": {…}}


def _process_raw_df(df) -> "pd.DataFrame":
    """Shared post-processing for raw DB or CSV data."""
    import pandas as pd
    df = df.copy()
    df["datum"]   = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    # Use category types to reduce memory
    for col in ("ort", "namn", "sort", "typ"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["_year"]   = df["datum"].dt.isocalendar().year.astype("int16")
    df["_week"]   = df["datum"].dt.isocalendar().week.astype("int8")
    df["_yw"]     = (df["_year"].astype(str) + "-W"
                     + df["_week"].astype(str).str.zfill(2)).astype("category")
    df["_day"]    = df["datum"].dt.day_name().astype("category")
    df["faktisk"] = (df["antal_ordrar"] - df["antal_returer"]).clip(lower=0).astype("int32")
    df.drop(columns=["antal_ordrar", "antal_returer"], errors="ignore", inplace=True)
    # Filter out paused products
    mask = df["sort"].astype(str).str.contains(
        r"beställ ej|Paus|\(EC\)", case=False, na=False)
    df = df[~mask]
    # Filter out non-stores
    store_sum  = df.groupby("namn")["faktisk"].sum()
    non_stores = store_sum[store_sum == 0].index.tolist() + ["Prover", "Alina Systems"]
    df = df[~df["namn"].isin(non_stores)]
    return df


def _get_raw_df():
    """Load historical data — DB first (when DB_HOST is set), fallback to 1year.csv.
    DB result is cached for DB_CACHE_SECONDS to avoid hitting the database on every request.
    """
    import time
    import pandas as pd

    # ── Primary: fetch from DB if configured ──────────────────
    if _DB_CFG:
        now  = time.time()
        last = _raw_state["db_fetched_at"]
        if _raw_state["df"] is not None and last and (now - last) < DB_CACHE_SECONDS:
            return _raw_state["df"]
        try:
            import sqlalchemy
            from urllib.parse import quote_plus
            pw  = quote_plus(str(_DB_CFG["password"]))
            url = (f"mysql+pymysql://{_DB_CFG['user']}:{pw}"
                   f"@{_DB_CFG['host']}:{_DB_CFG['port']}/{_DB_CFG['database']}?charset=utf8mb4")
            engine = sqlalchemy.create_engine(url, pool_pre_ping=True)
            try:
                # Apply category dtypes at read time to minimise peak memory usage
                raw = pd.read_sql(
                    """SELECT datum, namn, ort, typ, sort, antal_ordrar, antal_returer
                       FROM ordrar_och_returer_looker_1y
                       WHERE datum >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)
                       ORDER BY datum""",
                    engine,
                    dtype={"namn": "category", "ort": "category",
                           "sort": "category", "typ": "category"})
            finally:
                engine.dispose()
            df = _process_raw_df(raw)
            _raw_state["df"]            = df
            _raw_state["db_fetched_at"] = now
            _hist_cache.clear()
            _range_cache.clear()
            print(f"✓ DB: loaded {len(df):,} rows")
            return df
        except Exception as e:
            print(f"⚠ DB load failed ({e}), falling back to file")

    # ── Fallback: read from 1year.csv (local dev / Railway without DB vars) ──
    _ensure_data_file()
    if not DATA_FILE.exists():
        return None
    try:
        mtime = DATA_FILE.stat().st_mtime
        if _raw_state["df"] is None or _raw_state["mtime"] != mtime:
            needed  = {"datum", "antal_ordrar", "antal_returer", "ort", "namn", "sort", "typ"}
            header  = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()
            usecols = [c for c in header if c in needed]
            dtype   = {c: "category" for c in usecols if c in {"ort", "namn", "sort", "typ"}}
            raw     = pd.read_csv(DATA_FILE, usecols=usecols, dtype=dtype)
            df = _process_raw_df(raw)
            _raw_state["df"]    = df
            _raw_state["mtime"] = mtime
            _hist_cache.clear()
            _range_cache.clear()
        return _raw_state["df"]
    except Exception:
        return None

def compute_week_actual(year_week: str):
    """
    Compute driver view and kitchen view for a given week from 1year.csv.
    Returns {"driver": {…}, "kitchen": {…}}, or None if no data.
    """
    if year_week in _hist_cache:
        return _hist_cache[year_week]

    df = _get_raw_df()
    if df is None:
        return None

    week_df = df[df["_yw"] == year_week].copy()
    if week_df.empty:
        return None

    # ── Driver view ──
    sd = week_df.groupby(["ort", "namn", "_day"]).agg(
        qty=("faktisk", "sum")).reset_index()
    pd_ = (week_df[week_df["faktisk"] > 0]
           .groupby(["ort", "namn", "_day", "sort"])
           .agg(qty=("faktisk", "sum")).reset_index())

    driver = {}
    for _, row in sd[sd["qty"] > 0].iterrows():
        ort, namn, day, qty = row["ort"], row["namn"], row["_day"], int(row["qty"])
        prods = pd_[(pd_["ort"] == ort) & (pd_["namn"] == namn) &
                    (pd_["_day"] == day)]["sort"].tolist()
        driver.setdefault(ort, {}).setdefault(namn, []).append({
            "day":       day,
            "day_label": day_label(year_week, day),
            "qty":       qty,
            "range":     str(qty),   # Actual data = exact value, no range
            "products":  " | ".join(prods),
            "n":         len(prods),
        })
    for ort in driver:
        for namn in driver[ort]:
            driver[ort][namn].sort(key=lambda x: DAY_ORDER.get(x["day"], 9))

    # ── Kitchen view ──
    cat = week_df.groupby(["_day", "typ", "sort"]).agg(
        qty=("faktisk", "sum"), stores=("namn", "nunique")).reset_index()
    cat = cat[cat["qty"] > 0]

    kitchen = {}
    for day, ddf in cat.groupby("_day"):
        ddf = ddf.sort_values("qty", ascending=False)
        kitchen[day] = {
            "label": day_label(year_week, day),
            "rows": [
                {"type": r["typ"], "product": r["sort"],
                 "qty": int(r["qty"]), "range": str(int(r["qty"])),
                 "stores": int(r["stores"])}
                for _, r in ddf.iterrows()
            ]
        }

    result = {"driver": driver, "kitchen": kitchen}
    _hist_cache[year_week] = result
    return result

def compute_date_range_actual(start_str: str, end_str: str):
    """
    Compute driver view and kitchen view from 1year.csv for any date range.
    Each entry uses the actual date as its label (e.g. 'Mån 9/3').
    Returns {"driver": {…}, "kitchen": {…}}, or None if no data.
    """
    cache_key = (start_str, end_str)
    if cache_key in _range_cache:
        return _range_cache[cache_key]

    import pandas as pd
    df = _get_raw_df()
    if df is None:
        return None
    try:
        s = pd.Timestamp(start_str)
        e = pd.Timestamp(end_str) + pd.Timedelta(days=1)  # include end date
    except Exception:
        return None

    rdf = df[(df["datum"] >= s) & (df["datum"] < e)].copy()
    if rdf.empty:
        return None

    # Generate date labels: 'Mån 9/3'
    rdf["_dstr"]  = rdf["datum"].apply(lambda d: f"{d.day}/{d.month}")
    rdf["_dlbl"]  = (rdf["_day"].map(DAY_SHORT).fillna("").astype(str)
                     + " " + rdf["_dstr"])
    rdf["_dsort"] = rdf["datum"].dt.date   # used for sorting

    # ── Driver view ──
    sd = (rdf.groupby(["ort", "namn", "_dsort", "_dlbl"])
             .agg(qty=("faktisk", "sum"))
             .reset_index())
    prd = (rdf[rdf["faktisk"] > 0]
              .groupby(["ort", "namn", "_dsort", "sort"])
              .agg(qty=("faktisk", "sum"))
              .reset_index())

    driver = {}
    for _, row in sd[sd["qty"] > 0].iterrows():
        ort, namn = str(row["ort"]), str(row["namn"])
        dk, label, qty = row["_dsort"], str(row["_dlbl"]), int(row["qty"])
        prods = prd[(prd["ort"] == row["ort"]) & (prd["namn"] == row["namn"]) &
                    (prd["_dsort"] == dk)]["sort"].tolist()
        driver.setdefault(ort, {}).setdefault(namn, []).append({
            "day": str(dk), "day_label": label, "qty": qty,
            "range": str(qty),
            "products": " | ".join(str(p) for p in prods),
            "n": len(prods),
        })
    for ort in driver:
        for namn in driver[ort]:
            driver[ort][namn].sort(key=lambda x: x["day"])

    # ── Kitchen view ──
    cat = (rdf.groupby(["_dsort", "_dlbl", "typ", "sort"])
              .agg(qty=("faktisk", "sum"), stores=("namn", "nunique"))
              .reset_index())
    cat = cat[cat["qty"] > 0].sort_values(["_dsort", "qty"], ascending=[True, False])

    kitchen = {}
    for _, row in cat.iterrows():
        dk = str(row["_dsort"])
        if dk not in kitchen:
            kitchen[dk] = {"label": str(row["_dlbl"]), "rows": []}
        kitchen[dk]["rows"].append({
            "type": str(row["typ"]), "product": str(row["sort"]),
            "qty": int(row["qty"]), "range": str(int(row["qty"])),
            "stores": int(row["stores"]),
        })

    result = {"driver": driver, "kitchen": kitchen}
    _range_cache[cache_key] = result
    return result


def _should_show_prediction(from_d_str: str, to_d_str: str) -> bool:
    """Determine whether prediction data should be shown.
    Only show predictions in the following two cases:
      1. The requested range overlaps with the current delivery week (this week)
      2. The requested range overlaps with the model's predicted week (W11, etc.)
    W12, W13 and further future weeks return False (no reliable prediction).
    """
    from datetime import date as _date
    try:
        s = _date.fromisoformat(from_d_str)
        e = _date.fromisoformat(to_d_str)
    except Exception:
        return False
    today = _date.today()
    if e < today:          # purely historical range
        return False

    # ① Show predictions only for future ranges (start date strictly after today)
    if s > today:
        return True

    # ② Model predicted week — only if the range STARTS within the predicted week.
    # Using "start within window" (not overlap) avoids false-positives when the
    # ISO week's Sunday coincidentally equals the delivery week's opening Sunday.
    pw = get_predicted_week()
    if pw:
        pm = week_monday(pw)
        if pm:
            pw_sun = (pm - timedelta(days=1)).date()
            pw_sat = (pm + timedelta(days=5)).date()
            if pw_sun <= s <= pw_sat:
                return True

    return False


def _pred_date_map(start_str: str, end_str: str):
    """If the date range overlaps with the predicted week, return (pred_mon, overlap_dates_set), otherwise None.
    eataway delivery week = Sunday(-1) to Saturday(+5), based on Monday.
    """
    from datetime import date as _date
    pred_week = get_predicted_week()
    if not pred_week:
        return None
    pred_mon = week_monday(pred_week)
    if not pred_mon:
        return None
    try:
        s = _date.fromisoformat(start_str)
        e = _date.fromisoformat(end_str)
    except Exception:
        return None
    # Delivery week: Sunday(offset=-1) to Saturday(offset=+5)
    pred_start_d = (pred_mon - timedelta(days=1)).date()   # Sunday
    pred_end_d   = (pred_mon + timedelta(days=5)).date()   # Saturday
    if e < pred_start_d or s > pred_end_d:
        return None
    # Only keep predicted week dates that fall within the requested range (Sunday=-1 to Saturday=+5)
    in_range = set()
    for offset in range(-1, 6):
        d = (pred_mon + timedelta(days=offset)).date()
        if s <= d <= e:
            in_range.add(d)
    return pred_mon, in_range


def get_driver_predicted_for_dates(start_str: str, end_str: str) -> dict:
    """Map model predictions by weekday to actual dates within the requested date range.
    Model output is a weekday-level pattern that can be applied to any week (this week, next week, etc.).
    For each weekday, use the first occurrence within the range.
    """
    from datetime import date as _date
    df = load_csv(OUTPUT_DIR / "driver_view_v7.csv")
    if df is None:
        return {}
    try:
        s = _date.fromisoformat(start_str)
        e = _date.fromisoformat(end_str)
    except Exception:
        return {}

    # Build day-name → actual date mapping (for each weekday, use first occurrence within range)
    day_to_date = {}
    for offset in range((e - s).days + 1):
        d = s + timedelta(days=offset)
        day_name = d.strftime("%A")   # "Monday", "Tuesday", ...
        if day_name not in day_to_date:
            day_to_date[day_name] = d

    if not day_to_date:
        return {}

    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    result = {}
    for route, rdf in df.groupby("Route"):
        stores = {}
        for store, sdf in rdf.groupby("Store"):
            days = []
            for _, row in sdf.sort_values("_ord").iterrows():
                dn = row["Day"]
                if dn not in day_to_date:
                    continue
                d = day_to_date[dn]
                label = f"{DAY_SHORT.get(dn, dn)} {d.day}/{d.month}"
                days.append({
                    "day": str(d), "day_label": label,
                    "qty": int(row["Total_Qty"]), "range": str(row["Range"]),
                    "products": str(row.get("Products", "")),
                    "n": int(row.get("N_Products", 0)),
                })
            if days:
                stores[str(store)] = days
        if stores:
            result[str(route)] = stores
    return result


def get_kitchen_predicted_for_dates(start_str: str, end_str: str) -> dict:
    """Map kitchen predictions by weekday to actual dates within the requested date range."""
    from datetime import date as _date
    df = load_csv(OUTPUT_DIR / "kitchen_view_v7.csv")
    if df is None:
        return {}
    try:
        s = _date.fromisoformat(start_str)
        e = _date.fromisoformat(end_str)
    except Exception:
        return {}

    day_to_date = {}
    for offset in range((e - s).days + 1):
        d = s + timedelta(days=offset)
        day_name = d.strftime("%A")
        if day_name not in day_to_date:
            day_to_date[day_name] = d

    if not day_to_date:
        return {}

    df["_ord"] = df["Day"].map(DAY_ORDER).fillna(9)
    df = df.sort_values(["_ord", "Qty"], ascending=[True, False])

    result = {}
    for day, ddf in df.groupby("Day"):
        if day not in day_to_date:
            continue
        d = day_to_date[day]
        label = f"{DAY_SHORT.get(day, day)} {d.day}/{d.month}"
        result[str(d)] = {
            "label": label,
            "rows": [
                {"type": str(r["Type"]), "product": str(r["Product"]),
                 "qty": int(r["Qty"]), "range": str(r["Range"]),
                 "stores": int(r["Stores"])}
                for _, r in ddf.iterrows()
            ]
        }
    return result


def get_flat_predicted(from_d: str, to_d: str) -> list:
    """Flatten predictions into a list of datum/ort/namn/typ/sort/qty rows.
    Parses the Products column of driver_view_v7.csv and gets typ from kitchen_view_v7.csv.
    """
    from datetime import date as _date
    df = load_csv(OUTPUT_DIR / "driver_view_v7.csv")
    if df is None:
        return []
    # Build sort→typ mapping (from kitchen view)
    df_kit = load_csv(OUTPUT_DIR / "kitchen_view_v7.csv")
    prod_type: dict = {}
    if df_kit is not None and "Product" in df_kit.columns and "Type" in df_kit.columns:
        for _, r in df_kit.iterrows():
            prod_type[str(r["Product"])] = str(r["Type"])
    try:
        s = _date.fromisoformat(from_d)
        e = _date.fromisoformat(to_d)
    except Exception:
        return []
    # Weekday → actual date mapping
    day_to_date: dict = {}
    for offset in range((e - s).days + 1):
        d = s + timedelta(days=offset)
        dn = d.strftime("%A")
        if dn not in day_to_date:
            day_to_date[dn] = d

    rows: list = []
    for _, row in df.iterrows():
        dn = row["Day"]
        if dn not in day_to_date:
            continue
        actual_date = day_to_date[dn]
        products_str = str(row.get("Products", ""))
        if not products_str or products_str == "nan":
            continue
        for part in products_str.split(" | "):
            part = part.strip()
            if not part:
                continue
            if ": " in part:
                prod_name, qty_str = part.rsplit(": ", 1)
                try:
                    qty = int(float(qty_str))
                except Exception:
                    qty = 1
            else:
                prod_name, qty = part, 1
            typ = prod_type.get(prod_name,
                  prod_name.split("/")[0] if "/" in prod_name else "")
            rows.append({
                "datum": str(actual_date),
                "ort":   str(row["Route"]),
                "namn":  str(row["Store"]),
                "typ":   typ,
                "sort":  prod_name,
                "qty":   qty,
            })
    rows.sort(key=lambda r: (r["datum"], r["ort"], r["namn"], r["sort"]))
    return rows


def get_flat_actual(from_d: str, to_d: str) -> list:
    """Extract datum/ort/namn/typ/sort/qty row list from historical 1year.csv."""
    import pandas as pd
    df = _get_raw_df()
    if df is None:
        return []
    try:
        s = pd.Timestamp(from_d)
        e = pd.Timestamp(to_d) + pd.Timedelta(days=1)
    except Exception:
        return []
    rdf = df[(df["datum"] >= s) & (df["datum"] < e) & (df["faktisk"] > 0)].copy()
    if rdf.empty:
        return []
    rdf = rdf.sort_values(["datum", "ort", "namn", "sort"])
    return [
        {
            "datum": row.datum.strftime("%Y-%m-%d"),
            "ort":   str(row.ort),
            "namn":  str(row.namn),
            "typ":   str(row.typ),
            "sort":  str(row.sort),
            "qty":   int(row.faktisk),
        }
        for row in rdf.itertuples()
    ]


# ── Google Sheets export ──────────────────────────────────────
def _get_gsheet_creds():
    """
    Read Google service account credentials.
    Priority order:
      1. Environment variable GOOGLE_SHEETS_CREDS (Railway cloud, JSON string)
      2. Local file credentials.json (local development)
    """
    import json as _json
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # ① Prefer reading from environment variable (Railway cloud)
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDS", "")
    if creds_json:
        info = _json.loads(creds_json)
        return Credentials.from_service_account_info(info, scopes=scopes)

    # ② Fall back to local file (local development)
    if CREDENTIALS_FILE.exists():
        return Credentials.from_service_account_file(str(CREDENTIALS_FILE), scopes=scopes)

    raise FileNotFoundError(
        "Google credentials not found: please set environment variable GOOGLE_SHEETS_CREDS, "
        f"or place credentials.json at {CREDENTIALS_FILE}"
    )


def export_to_gsheet(from_d: str = "", to_d: str = "") -> dict:
    """Write predicted Tabell data (datum/ort/namn/typ/sort/qty) to Google Sheet (overwrite)."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials  # noqa: F401
    except ImportError:
        return {"ok": False, "msg": "Missing dependencies, please run: pip install gspread google-auth"}

    # If no date specified, automatically use the current predicted week (Sunday → Saturday)
    if not from_d or not to_d:
        pred_week = get_predicted_week()
        if pred_week:
            mon = week_monday(pred_week)
            if mon:
                from_d = (mon - timedelta(days=1)).strftime("%Y-%m-%d")
                to_d   = (mon + timedelta(days=5)).strftime("%Y-%m-%d")
    if not from_d or not to_d:
        return {"ok": False, "msg": "Predicted week not found, cannot export"}

    rows = get_flat_predicted(from_d, to_d)
    if not rows:
        return {"ok": False, "msg": "No prediction data to export"}

    try:
        creds = _get_gsheet_creds()
        gc    = gspread.authorize(creds)
        sh    = gc.open_by_key(GSHEET_ID)

        # Get or create worksheet Tab
        try:
            ws = sh.worksheet(GSHEET_TAB)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=GSHEET_TAB, rows=5000, cols=10)

        # Clear and overwrite
        ws.clear()
        header = [["Datum", "Ort", "Butik", "Typ", "Produkt", "Antal"]]
        data   = header + [
            [r["datum"], r["ort"], r["namn"], r["typ"], r["sort"], r["qty"]]
            for r in rows
        ]
        ws.update(data, "A1")

        date_label = fmt_date_range(from_d, to_d)
        return {"ok": True, "msg": f"Wrote {len(rows)} rows ({date_label}) → Google Sheet ✓"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def get_all_weeks():
    """Return all available weeks (actual + predicted)."""
    weeks = {}
    df = _get_raw_df()
    if df is not None:
        for yw in df["_yw"].unique():
            weeks[yw] = "actual"
    pred = get_predicted_week()
    if pred:
        # Mark the model output week and beyond as predicted
        if pred not in weeks:
            weeks[pred] = "predicted"
        else:
            weeks[pred] = "predicted"   # override (recent weeks have incomplete data)
    return sorted([{"week": k, "type": v} for k, v in weeks.items()],
                  key=lambda x: x["week"])


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    has_output = (OUTPUT_DIR / "driver_view_v7.csv").exists()
    metrics    = get_metrics() if has_output else None
    pred_week  = get_predicted_week() if has_output else ""

    # V7.6 Change: Default date range is current week (Sun-Sat).
    # If today is Friday or Saturday, default to next week.
    today = datetime.now()
    if today.weekday() in [4, 5]: # Friday or Saturday
        base_date = today + timedelta(weeks=1)
    else:
        base_date = today
    days_since_sunday = (base_date.weekday() + 1) % 7
    init_from_dt = base_date - timedelta(days=days_since_sunday)
    init_to_dt = init_from_dt + timedelta(days=6)
    init_from = init_from_dt.strftime("%Y-%m-%d")
    init_to = init_to_dt.strftime("%Y-%m-%d")

    # Earliest selectable date = earliest date in historical data
    df_raw   = _get_raw_df()
    min_date = ""
    if df_raw is not None and "datum" in df_raw.columns:
        min_date = df_raw["datum"].min().strftime("%Y-%m-%d")

    return render_template("index.html",
                           has_output=has_output,
                           init_from=init_from,
                           init_to=init_to,
                           date_range=fmt_date_range(init_from, init_to),
                           metrics=metrics,
                           min_date=min_date,
                           pw_required=bool(APP_PASSWORD))

@app.route("/api/weeks")
def api_weeks():
    return jsonify(get_all_weeks())

@app.route("/api/flat")
def api_flat():
    """Return flat table data: datum/ort/namn/typ/sort/qty row list."""
    from_d = request.args.get("from", "").strip()
    to_d   = request.args.get("to",   "").strip()
    if not from_d or not to_d:
        return jsonify({"data": [], "date_range": "", "type": "nodata"})
    date_range_label = fmt_date_range(from_d, to_d)
    is_pred = _should_show_prediction(from_d, to_d)
    if is_pred:
        rows = get_flat_predicted(from_d, to_d)
        if rows:
            return jsonify({"data": rows, "date_range": date_range_label, "type": "predicted"})
    rows = get_flat_actual(from_d, to_d)
    if rows:
        return jsonify({"data": rows, "date_range": date_range_label, "type": "actual"})
    rows = get_flat_predicted(from_d, to_d)
    if rows:
        return jsonify({"data": rows, "date_range": date_range_label, "type": "predicted"})
    return jsonify({"data": [], "date_range": date_range_label, "type": "nodata"})

@app.route("/api/download/flat")
def api_download_flat():
    """Download flat-format CSV: datum/ort/namn/typ/sort/qty."""
    from_d = request.args.get("from", "").strip()
    to_d   = request.args.get("to",   "").strip()
    if not from_d or not to_d:
        return "Missing parameters", 400
    is_pred = _should_show_prediction(from_d, to_d)
    if is_pred:
        rows = get_flat_predicted(from_d, to_d) or get_flat_actual(from_d, to_d)
    else:
        rows = get_flat_actual(from_d, to_d) or get_flat_predicted(from_d, to_d)
    if not rows:
        return "Ingen data", 404
    import io, csv as csv_mod
    from flask import Response
    buf = io.StringIO()
    writer = csv_mod.DictWriter(buf, fieldnames=["datum","ort","namn","typ","sort","qty"])
    writer.writeheader()
    writer.writerows(rows)
    csv_bytes = ("\ufeff" + buf.getvalue()).encode("utf-8")   # BOM for Excel compatibility
    filename = f"eataway_{from_d}_{to_d}.csv"
    return Response(csv_bytes, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})

@app.route("/api/driver")
def api_driver():
    from_d = request.args.get("from", "").strip()
    to_d   = request.args.get("to",   "").strip()
    week   = request.args.get("week", "").strip()
    pred   = get_predicted_week()

    # ── Date range mode (used by new UI) ─────────────────────────────
    if from_d and to_d:
        date_range_label = fmt_date_range(from_d, to_d)
        # Only show predictions for current week or predicted week, to avoid W12+ all returning the same data
        is_future = _should_show_prediction(from_d, to_d)

        if is_future:
            # Contains future dates → prefer returning model predictions (mapped by weekday to requested dates)
            pred_data = get_driver_predicted_for_dates(from_d, to_d)
            if pred_data:
                return jsonify({
                    "data":       pred_data,
                    "date_range": date_range_label,
                    "type":       "predicted",
                })
        # Purely historical dates → look up actual data
        actual = compute_date_range_actual(from_d, to_d)
        if actual is not None:
            return jsonify({
                "data":       actual["driver"],
                "date_range": date_range_label,
                "type":       "actual",
            })
        # No historical data, also try predictions
        pred_data = get_driver_predicted_for_dates(from_d, to_d)
        if pred_data:
            return jsonify({
                "data":       pred_data,
                "date_range": date_range_label,
                "type":       "predicted",
            })
        return jsonify({"data": {}, "date_range": date_range_label, "type": "nodata"})

    # ── Week mode (backward compatible with old parameters) ───────────
    if not week:
        week = pred

    # Prefer historical actual data first
    actual = compute_week_actual(week) if week else None
    if actual is not None:
        return jsonify({
            "data":       actual["driver"],
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "actual",
        })

    # No actual data → only return predictions for "model predicted week", return nodata for other weeks (avoids all future weeks showing the same data)
    if pred and week == pred:
        return jsonify({
            "data":       get_driver_predicted(display_yw=week),
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "predicted",
        })

    return jsonify({"data": {}, "year_week": week,
                    "date_range": week_to_range(week), "type": "nodata"})

@app.route("/api/kitchen")
def api_kitchen():
    from_d = request.args.get("from", "").strip()
    to_d   = request.args.get("to",   "").strip()
    week   = request.args.get("week", "").strip()
    pred   = get_predicted_week()

    # ── Date range mode ───────────────────────────────────────────────
    if from_d and to_d:
        date_range_label = fmt_date_range(from_d, to_d)
        is_future = _should_show_prediction(from_d, to_d)

        if is_future:
            pred_data = get_kitchen_predicted_for_dates(from_d, to_d)
            if pred_data:
                return jsonify({
                    "data":       pred_data,
                    "date_range": date_range_label,
                    "type":       "predicted",
                })
        actual = compute_date_range_actual(from_d, to_d)
        if actual is not None:
            return jsonify({
                "data":       actual["kitchen"],
                "date_range": date_range_label,
                "type":       "actual",
            })
        pred_data = get_kitchen_predicted_for_dates(from_d, to_d)
        if pred_data:
            return jsonify({
                "data":       pred_data,
                "date_range": date_range_label,
                "type":       "predicted",
            })
        return jsonify({"data": {}, "date_range": date_range_label, "type": "nodata"})

    # ── Week mode (backward compatible) ───────────────────────────────
    if not week:
        week = pred

    actual = compute_week_actual(week) if week else None
    if actual is not None:
        return jsonify({
            "data":       actual["kitchen"],
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "actual",
        })

    if pred and week == pred:
        return jsonify({
            "data":       get_kitchen_predicted(display_yw=week),
            "year_week":  week,
            "date_range": week_to_range(week),
            "type":       "predicted",
        })

    return jsonify({"data": {}, "year_week": week,
                    "date_range": week_to_range(week), "type": "nodata"})

@app.route("/api/run", methods=["POST"])
def api_run():
    if not _password_ok():
        return jsonify({"ok": False, "msg": "Invalid password"}), 403
    if pipeline_status["running"]:
        return jsonify({"ok": False, "msg": "Pipeline already running"})
    threading.Thread(target=run_pipeline_thread, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify({
        "running": pipeline_status["running"],
        "done":    pipeline_status["done"],
        "error":   pipeline_status["error"],
        "log":     pipeline_status["log"][-200:],
    })

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if not _password_ok():
        return jsonify({"ok": False, "msg": "Invalid password"}), 403
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "msg": "No file selected"})
    fname = f.filename.lower()
    if not (fname.endswith(".csv") or fname.endswith(".zip")):
        return jsonify({"ok": False, "msg": "Only CSV or ZIP files accepted"})
    try:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        if fname.endswith(".zip"):
            # Save zip and extract 1year.csv
            import zipfile, io
            data = f.read()
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                names = zf.namelist()
                csv_name = next((n for n in names if n.endswith(".csv")), None)
                if not csv_name:
                    return jsonify({"ok": False, "msg": "No CSV file found in ZIP"})
                with zf.open(csv_name) as src, open(DATA_FILE, "wb") as dst:
                    dst.write(src.read())
        else:
            f.save(str(DATA_FILE))
        # Invalidate cache
        _raw_state["df"] = None
        _hist_cache.clear()
        _range_cache.clear()
        size_mb = DATA_FILE.stat().st_size / 1024 / 1024
        return jsonify({"ok": True, "msg": f"Upload successful ({size_mb:.1f} MB), historical data updated"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/upload-results", methods=["POST"])
def api_upload_results():
    """Receive a locally-trained prediction results ZIP and extract it to overwrite output_v7/ and predictions/."""
    if not _password_ok():
        return jsonify({"ok": False, "msg": "Invalid password"}), 403
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "msg": "No file selected"})
    try:
        import zipfile, io as _io
        data = f.read()
        with zipfile.ZipFile(_io.BytesIO(data), "r") as zf:
            zf.extractall(BASE_DIR)
        # Clear all caches so new data takes effect immediately
        _raw_state["df"] = None
        _hist_cache.clear()
        _range_cache.clear()
        return jsonify({"ok": True, "msg": "Prediction results updated ✓"})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})


@app.route("/api/diagnostics")
def api_diagnostics():
    path = OUTPUT_DIR / "diagnostics_v7.png"
    if path.exists():
        return send_file(str(path), mimetype="image/png")
    return "No diagnostics", 404

@app.route("/api/download/<name>")
def api_download(name):
    allowed = {"driver": "driver_view_v7.csv", "kitchen": "kitchen_view_v7.csv"}
    if name not in allowed:
        return "Not found", 404
    path = OUTPUT_DIR / allowed[name]
    if not path.exists():
        return "File not found", 404
    return send_file(str(path), as_attachment=True)

@app.route("/api/export-gsheet", methods=["POST"])
def api_export_gsheet():
    """Manual trigger: export Tabell prediction data to Google Sheet."""
    body   = request.get_json(silent=True) or {}
    from_d = body.get("from", "")
    to_d   = body.get("to",   "")
    result = export_to_gsheet(from_d, to_d)
    return jsonify(result)


# ── Startup ───────────────────────────────────────────────────
if __name__ == "__main__":
    port     = int(os.environ.get("PORT", 5000))
    is_local = not os.environ.get("PORT")
    print("\n" + "=" * 50)
    print(f"  Eataway  http://127.0.0.1:{port}")
    if APP_PASSWORD:
        print("  Password protection: enabled")
    print("=" * 50 + "\n")
    if is_local:
        import threading, webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=False)
