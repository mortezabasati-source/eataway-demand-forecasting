import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import json
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ============================================================================
# Config
OUTPUT_DIR = Path(__file__).parent / "output_v7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)# ============================================================================
DATA_PATH    = Path(__file__).parent / "trainable_data.csv"
WEATHER_PATH = str(Path(__file__).parent / "weather_weekly.csv")


SEED = 42

TARGET = "faktisk"
ID_COLS = ["namn", "ort", "typ", "sort", "product_id", "year", "week", "year_week"]

BASE_FEATURES = [
    "week", "month", "week_sin", "week_cos", "month_sin", "month_cos",
    "is_december", "is_summer",
    "is_holiday_week", "n_holidays", "is_high_impact_holiday",
    "is_pre_holiday_week", "is_post_holiday_week",
    "lag_1w", "lag_2w", "lag_3w", "lag_4w",
    "rolling_mean_4w", "rolling_mean_8w", "rolling_mean_12w",
    "rolling_std_4w", "rolling_std_8w",
    "rolling_median_4w", "rolling_max_4w",
    # rolling_min_4w removed: anchors predictions to historical lows, causing systematic underestimation
    "trend_4w", "yoy_same_week",
    "return_rate_lag1", "rolling_return_rate_4w", "censored_ratio_4w",
    "store_avg_weekly", "store_std_weekly", "store_n_products",
    "store_product_share", "route_avg_demand",
    "product_avg_weekly", "product_std_weekly", "product_n_stores",
    "category_season_avg", "category_month_avg", "category_holiday_avg",
    "namn_te", "product_id_te", "typ_te", "ort_te",
    "season_winter", "season_spring", "season_summer", "season_autumn",
]

ORT_TO_CITY = {
    "Gävle": "Gävle",
    "Stockholm Mellan": "Stockholm", "Stockholm N": "Stockholm",
    "Stockholm Södra": "Stockholm", "Stockholm Väst": "Stockholm",
    "Stockholm Xtra": "Stockholm", "Stockholm Östra": "Stockholm",
    "Uppsala Rutt": "Uppsala", "Västerås": "Västerås",
}


# ============================================================================
# P0: Incomplete week detection
# ============================================================================
def detect_incomplete_weeks(df, threshold=0.7):
    print("=" * 70)
    print("P0: Detecting incomplete data weeks")
    print("=" * 70)
    weekly = df[df[TARGET] > 0].groupby("year_week").agg(
        stores=("namn", "nunique"), demand=(TARGET, "sum"))
    all_wks = sorted(df["year_week"].unique())
    weekly = weekly.reindex(all_wks)
    bad = []
    for i, wk in enumerate(all_wks):
        if i < 4:
            continue
        c = weekly.loc[wk]
        p = weekly.iloc[max(0, i-4):i]
        sr = c["stores"] / max(p["stores"].mean(), 1)
        dr = c["demand"] / max(p["demand"].mean(), 1)
        if sr < threshold or dr < threshold:
            bad.append(wk)
            print(f"  Warning {wk}: store_ratio={sr:.0%} demand_ratio={dr:.0%}")
    if not bad:
        print("  OK no incomplete weeks")
    print()
    return bad


# ============================================================================
# Weather features
# ============================================================================
def merge_weather(df):
    print("=" * 70)
    print("Weather feature merge")
    print("=" * 70)

    wp = Path(WEATHER_PATH)
    if not wp.exists():
        print(f"  Warning: {WEATHER_PATH} not found")
        print("  -> Skip weather (run eataway_weather.py first)")
        print()
        return df, []

    weather = pd.read_csv(wp)
    print(f"  Weather: {len(weather)} rows, {weather['city'].nunique()} cities")
    print(f"  Range: {weather['year_week'].min()} ~ {weather['year_week'].max()}")

    df["weather_city"] = df["ort"].map(ORT_TO_CITY).fillna("Stockholm")

    weather_cols = [
        "temp_mean", "temp_max", "temp_min", "temp_range",
        "precip_total", "rain_total", "snow_total",
        "wind_max", "gust_max", "sunshine_hrs",
        "rain_days", "snow_days",
        "is_rainy_week", "is_snowy_week", "is_cold", "is_hot",
    ]
    avail = [c for c in weather_cols if c in weather.columns]

    wr = weather.rename(columns={"city": "weather_city"})[["weather_city", "year_week"] + avail]
    before = len(df)
    df = df.merge(wr, on=["weather_city", "year_week"], how="left")
    assert len(df) == before

    matched = df[avail[0]].notna().sum()
    print(f"  Match rate: {matched}/{len(df)} ({matched/len(df):.1%})")

    for c in avail:
        df[c] = df[c].fillna(df[c].median())

    extra = []

    # Temperature anomaly (deviation from monthly mean in degrees)
    if "temp_mean" in df.columns and "month" in df.columns:
        mm = df.groupby("month")["temp_mean"].transform("mean")
        df["temp_anomaly"] = df["temp_mean"] - mm
        extra.append("temp_anomaly")

    # Bad weather composite score
    if "precip_total" in df.columns and "wind_max" in df.columns:
        df["bad_weather"] = (
            (df["precip_total"] > df["precip_total"].quantile(0.75)).astype(int) +
            (df["wind_max"] > df["wind_max"].quantile(0.75)).astype(int) +
            df.get("is_snowy_week", pd.Series(0, index=df.index)).astype(int)
        )
        extra.append("bad_weather")

    # Last week's weather (lagged)
    for col in ["temp_mean", "precip_total"]:
        if col in df.columns:
            lc = f"{col}_lag1w"
            df[lc] = df.groupby(["namn", "product_id"])[col].shift(1)
            df[lc] = df[lc].fillna(df[col].median())
            extra.append(lc)

    all_wf = avail + extra
    print(f"  Weather features: {len(all_wf)}")
    print()
    return df, all_wf


# ============================================================================
# Demand features
# ============================================================================
def add_demand_features(df):
    print("=" * 70)
    print("Demand feature engineering")
    print("=" * 70)
    df = df.sort_values(["namn", "product_id", "year", "week"]).reset_index(drop=True)
    g = ["namn", "product_id"]
    new = []

    df["recent_zero_count"] = df.groupby(g)[TARGET].transform(
        lambda x: (x.shift(1) == 0).rolling(4, min_periods=1).sum())
    new.append("recent_zero_count")

    def wsn(x):
        r = pd.Series(index=x.index, dtype=float)
        last = -1
        for i in range(len(x)):
            if i > 0 and x.iloc[i-1] > 0:
                last = i - 1
            r.iloc[i] = (i - last) if last >= 0 else 8
        return r

    df["weeks_since_nonzero"] = df.groupby(g)[TARGET].transform(wsn).clip(0, 12)
    new.append("weeks_since_nonzero")

    df["active_ratio_8w"] = df.groupby(g)[TARGET].transform(
        lambda x: (x.shift(1) > 0).rolling(8, min_periods=2).mean()).fillna(0.5)
    new.append("active_ratio_8w")

    def nz_mean(x, w=4):
        r = pd.Series(index=x.index, dtype=float)
        s = x.shift(1)
        for i in range(len(x)):
            start = max(0, i-w)
            v = s.iloc[start:i]
            nz = v[v > 0]
            r.iloc[i] = nz.mean() if len(nz) > 0 else 0
        return r

    df["nonzero_mean_4w"] = df.groupby(g)[TARGET].transform(lambda x: nz_mean(x, 4))
    new.append("nonzero_mean_4w")

    rm4 = df.get("rolling_mean_4w", pd.Series(0, index=df.index))
    rs4 = df.get("rolling_std_4w", pd.Series(0, index=df.index))
    df["demand_cv_4w"] = np.where(rm4 > 0, rs4 / rm4.clip(lower=0.1), 0)
    df["demand_cv_4w"] = df["demand_cv_4w"].clip(0, 5).fillna(0)
    new.append("demand_cv_4w")

    df["lag1_zscore"] = np.where(
        rs4 > 0,
        (df.get("lag_1w", pd.Series(0, index=df.index)) - rm4) / rs4.clip(lower=0.1), 0)
    df["lag1_zscore"] = pd.to_numeric(df["lag1_zscore"], errors="coerce").fillna(0).clip(-5, 5)
    new.append("lag1_zscore")

    df["is_positive"] = (df[TARGET] > 0).astype(int)
    df["log_target"] = np.log1p(df[TARGET])

    print(f"  Demand features: {len(new)}")
    print()
    return df, new


# ============================================================================
# Load
# ============================================================================
def load_and_prepare():
    print("\n" + "=" * 70)
    print("  EATAWAY V4 — Data Preparation")
    print("=" * 70 + "\n")

    df_all = pd.read_csv(DATA_PATH)
    print(f"  Raw: {len(df_all):,} rows\n")

    # Prefer is_truncated column flagged by feature.py; detect manually if absent
    if "is_truncated" in df_all.columns:
        # CSV may read back as str/float/NaN — normalize to bool
        df_all["is_truncated"] = (
            df_all["is_truncated"]
            .fillna(False)
            .apply(lambda x: str(x).strip().lower() in ("true", "1", "1.0"))
        )
        truncated_weeks = df_all[df_all["is_truncated"]]["year_week"].unique().tolist()
        print(f"  Truncated weeks (from feature.py): {truncated_weeks}")
        df = df_all[~df_all["is_truncated"]].copy()
    else:
        bad = detect_incomplete_weeks(df_all)
        truncated_weeks = bad
        df = df_all[~df_all["year_week"].isin(bad)].copy() if bad else df_all.copy()

    # Save truncated weeks separately for out-of-sample prediction after training
    df_holdout = df_all[df_all["year_week"].isin(truncated_weeks)].copy() if truncated_weeks else pd.DataFrame()
    print(f"  Available for training: {len(df):,} rows | Out-of-sample (truncated weeks): {len(df_holdout):,} rows\n")

    df, wf = merge_weather(df)
    df, df_feats = add_demand_features(df)

    # Apply same demand features to holdout (fill with training-set statistics)
    if not df_holdout.empty:
        df_holdout, _ = merge_weather(df_holdout)
        df_holdout, _ = add_demand_features(df_holdout)

    features = list(dict.fromkeys(
        [c for c in BASE_FEATURES + df_feats + wf if c in df.columns]))
    print(f"  Final: {len(df):,} rows, {len(features)} features\n")
    return df, features, df_holdout


def time_split(df, test_w=6, val_w=6):
    print("=" * 70)
    print("Time Split")
    print("=" * 70)
    wks = (df[["year","week","year_week"]].drop_duplicates()
           .sort_values(["year","week"]).reset_index(drop=True))
    n = len(wks)
    ts, vs = n - test_w, n - test_w - val_w
    tw = wks.iloc[:vs]["year_week"].tolist()
    vw = wks.iloc[vs:ts]["year_week"].tolist()
    tew = wks.iloc[ts:]["year_week"].tolist()
    dtr = df[df["year_week"].isin(tw)].copy()
    dva = df[df["year_week"].isin(vw)].copy()
    dte = df[df["year_week"].isin(tew)].copy()
    for nm, d, w in [("Train",dtr,tw),("Val",dva,vw),("Test",dte,tew)]:
        print(f"  {nm}: {len(d):>8,} rows | {len(w)} weeks | {w[0]}~{w[-1]}")
    print()
    return dtr, dva, dte


# ============================================================================
# V4 Core: CalibratedHurdleModel
# ============================================================================

class CalibratedHurdleModel:

    def __init__(self):
        self.cls_model = None
        self.reg_model = None
        self.calibrator = None
        self.threshold = 0.30
        self.bias_factors = None

    def fit(self, df_train, df_val, features):
        X_tr, X_va = df_train[features], df_val[features]

        # Stage 1: Classifier
        print("  -- Stage 1: Classifier --")
        y1t = df_train["is_positive"]
        y1v = df_val["is_positive"]
        p1 = {"objective":"binary","metric":"binary_logloss",
              "learning_rate":0.05,"num_leaves":31,"max_depth":6,
              "min_child_samples":30,"subsample":0.8,"colsample_bytree":0.8,
              "reg_alpha":0.5,"reg_lambda":2.0,"is_unbalance":True,
              "n_jobs":-1,"seed":SEED,"verbose":-1}
        d1 = lgb.Dataset(X_tr, label=y1t)
        d1v = lgb.Dataset(X_va, label=y1v, reference=d1)
        self.cls_model = lgb.train(
            p1, d1, num_boost_round=3000, valid_sets=[d1v],
            callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])

        # Isotonic calibration
        pr = self.cls_model.predict(X_va)
        self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.calibrator.fit(pr, y1v.values)
        pc = self.calibrator.predict(pr)

        print(f"    best_iter={self.cls_model.best_iteration}")
        low_mask = pr < 0.1
        if low_mask.sum() > 0:
            print(f"    Before cal: P<0.1 actual positive rate = {y1v.values[low_mask].mean():.0%}")
            print(f"    After cal:  P<0.1 -> calibrated P mean = {pc[low_mask].mean():.2f}")
        print()

        # Stage 2: Regressor (positives only, log-space)
        print("  -- Stage 2: Regressor --")
        pos_tr = df_train[df_train["is_positive"] == 1]
        pos_va = df_val[df_val["is_positive"] == 1]
        y2t = np.log1p(pos_tr[TARGET].values)
        y2v = np.log1p(pos_va[TARGET].values)
        p2 = {"objective":"regression_l1","metric":"mae",
              "learning_rate":0.05,"num_leaves":63,"min_child_samples":20,
              "subsample":0.8,"colsample_bytree":0.8,
              "reg_alpha":0.1,"reg_lambda":1.0,
              "n_jobs":-1,"seed":SEED,"verbose":-1}
        d2 = lgb.Dataset(pos_tr[features], label=y2t)
        d2v = lgb.Dataset(pos_va[features], label=y2v, reference=d2)
        self.reg_model = lgb.train(
            p2, d2, num_boost_round=5000, valid_sets=[d2v],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        print(f"    best_iter={self.reg_model.best_iteration}")

        # Lognormal correction: E[exp(X)] = exp(μ + σ²/2) not exp(μ)
        # Larger log-space residual variance → more severe underestimation on back-transform → larger correction needed
        lp_val = self.reg_model.predict(pos_va[features])
        self.log_sigma2 = float(np.var(y2v - lp_val))
        ln_factor = np.exp(min(self.log_sigma2 / 2, 0.5))  # cap at e^0.5 ≈ 1.65x
        print(f"    log_sigma2={self.log_sigma2:.3f}  lognormal_factor={ln_factor:.3f}x")
        print()

    def _raw_predict(self, X):
        pr = self.cls_model.predict(X)
        pc = self.calibrator.predict(pr)
        lp = self.reg_model.predict(X)
        # Apply lognormal correction: compensate for systematic underestimation due to Jensen's inequality
        sigma2 = getattr(self, "log_sigma2", 0.0)
        lp_corrected = lp + min(sigma2 / 2, 0.5)  # same cap as during training
        rp = np.clip(np.expm1(lp_corrected), 0, None)

        # V4: hard gate + no multiplication by P
        y = rp.copy()
        y[pc < self.threshold] = 0
        return y, pc, rp

    def optimize_threshold(self, df_val, features):
        print("  -- Threshold optimization --")
        X = df_val[features]
        yt = df_val[TARGET].values
        pr = self.cls_model.predict(X)
        pc = self.calibrator.predict(pr)
        lp = self.reg_model.predict(X)
        rp = np.clip(np.expm1(lp), 0, None)

        best_score, best_t = float("inf"), 0.30
        print(f"    {'Thr':>5s}  {'MAE':>6s}  {'Bias':>7s}  {'Zero%':>6s}  {'FN':>5s}  {'Score':>7s}")
        print(f"    {'-'*46}")

        for t in np.arange(0.40, 0.85, 0.05): # Ändrat startvärde från 0.15 till 0.40
            yp = rp.copy()
            yp[pc < t] = 0
            yp = np.round(yp)
            mae  = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            zr   = (yp == 0).mean()
            fn   = ((yt > 0) & (yp == 0)).sum()
            # Combined score: MAE + missed-positive penalty (underestimation cost is 2x overestimation)
            # This pushes the optimizer toward a threshold with bias near zero or slightly positive, reducing missed positives
            score = mae + 0.5 * max(0.0, -bias)
            mk = " <" if score < best_score else ""
            if score < best_score:
                best_score, best_t = score, t
            print(f"    {t:5.2f}  {mae:6.3f}  {bias:+7.3f}  {zr:5.1%}  {fn:>5d}  {score:7.3f}{mk}")

        self.threshold = best_t
        print(f"    Best: {best_t:.2f} (score={best_score:.3f})")
        print()

    def learn_bias_correction(self, df_val, features):
        """
        Stratified bias correction — based on predicted value bins (also available at inference time)

        V7 changes:
          - Narrower clip range: 0.5~2.0 → 0.8~1.3 (prevent overcorrection)
          - For y>=6 bin, only allow upward correction (f>=1.0), no downward compression
            Root cause: high-demand compression on val set → same underestimation on test set
          - Minimum sample size raised from 20 to 30
        """
        print("  -- Bias correction learning --")
        X  = df_val[features]
        yt = df_val[TARGET].values
        yr, _, _ = self._raw_predict(X)
        yr = np.round(yr)

        bins = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 999)]
        self.bias_factors = {}

        for lo, hi in bins:
            mask = (yr >= lo) & (yr <= hi)
            if mask.sum() < 30:
                self.bias_factors[(lo, hi)] = 1.0
                continue
            pm = yr[mask].mean()
            tm = yt[mask].mean()
            f  = tm / max(pm, 0.01) if pm > 0.1 else 1.0

            # Wider upper bound: corrections supported by val data should be used
            # The y=3-5 bin mixes in many underestimated y_true=6-10 samples,
            # causing tm >> pm in that bin, requiring correction factor above 1.3x
            if lo >= 6:
                f = np.clip(f, 1.0, 1.15)   # Sänkt från 2.2 för att minska aggressiv uppjustering
            else:
                f = np.clip(f, 0.8, 1.25)   # Justerat för en mer balanserad korrigering

            self.bias_factors[(lo, hi)] = f
            direction = "↑" if f > 1.0 else ("↓" if f < 1.0 else "=")
            print(f"    pred={lo}-{hi:>3d}: pred={pm:.2f} true={tm:.2f} f={f:.3f} {direction}")
        print()

    def predict(self, X):
        yr, pc, rp = self._raw_predict(X)
        yc = yr.copy()

        if self.bias_factors:
            for (lo, hi), f in self.bias_factors.items():
                mask = (yr >= lo) & (yr <= hi)
                if mask.any():
                    yc[mask] = yr[mask] * f

        yf = np.clip(np.round(yc), 0, None).astype(int)
        # yc is bias-corrected float (pre-rounding), used by gen_views before applying global_scale
        return yf, pc, rp, yc

    def feature_importance(self, features):
        cg = self.cls_model.feature_importance(importance_type="gain")
        rg = self.reg_model.feature_importance(importance_type="gain")
        fi = pd.DataFrame({"feature": features,
                           "cls": cg / max(cg.sum(),1) * 100,
                           "reg": rg / max(rg.sum(),1) * 100})
        fi["combined"] = fi["cls"] * 0.3 + fi["reg"] * 0.7
        return fi.sort_values("combined", ascending=False).reset_index(drop=True)


# ============================================================================
# Tweedie
# ============================================================================
class TweedieModel:
    def __init__(self):
        self.model = None

    def fit(self, df_train, df_val, features):
        print("  -- Tweedie --")
        p = {"objective":"tweedie","tweedie_variance_power":1.5,
             "metric":"mae","learning_rate":0.05,"num_leaves":63,
             "min_child_samples":20,"subsample":0.8,"colsample_bytree":0.8,
             "reg_alpha":0.1,"reg_lambda":1.0,"n_jobs":-1,"seed":SEED,"verbose":-1}
        d = lgb.Dataset(df_train[features], label=df_train[TARGET].values.astype(float))
        dv = lgb.Dataset(df_val[features], label=df_val[TARGET].values.astype(float), reference=d)
        self.model = lgb.train(
            p, d, num_boost_round=5000, valid_sets=[dv],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        print(f"    best_iter={self.model.best_iteration}")
        print()

    def predict(self, X):
        return np.clip(self.model.predict(X), 0, None)


# ============================================================================
# V4 Ensemble
# ============================================================================
class V4Ensemble:
    def __init__(self):
        self.hurdle = CalibratedHurdleModel()
        self.tweedie = TweedieModel()
        self.weights = [0.65, 0.35]
        self.features = None

    def fit(self, df_train, df_val, features):
        self.features = features
        print("=" * 70)
        print("V4 Training")
        print("=" * 70)

        self.hurdle.fit(df_train, df_val, features)
        self.hurdle.optimize_threshold(df_val, features)
        self.hurdle.learn_bias_correction(df_val, features)
        self.tweedie.fit(df_train, df_val, features)
        self._opt_weights(df_val, features)

    def _opt_weights(self, df_val, features):
        print("  -- Ensemble weights --")
        X = df_val[features]
        yt = df_val[TARGET].values
        ph, _, _, _ = self.hurdle.predict(X)
        pt = self.tweedie.predict(X)

        best_score, best_w = float("inf"), 0.65
        for wh in np.arange(0.3, 0.9, 0.05):
            wt = 1.0 - wh
            yp  = np.clip(np.round(wh*ph + wt*pt), 0, None)
            mae  = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            # Same scoring as threshold optimization: underestimation cost is 2x overestimation
            score = mae + 0.5 * max(0.0, -bias)
            if score < best_score:
                best_score, best_w = score, wh

        self.weights = [best_w, 1.0 - best_w]
        print(f"    Hurdle={best_w:.0%} Tweedie={1-best_w:.0%} Score={best_score:.3f}")

        for nm, pred in [("Hurdle+BiasCorr", ph), ("Tweedie", pt)]:
            yp = np.clip(np.round(pred.astype(float)), 0, None)
            mae = np.mean(np.abs(yt - yp))
            bias = np.mean(yp - yt)
            print(f"    {nm:20s}: MAE={mae:.3f} Bias={bias:+.3f}")
        print()

    def predict(self, X):
        ph, pc, rp, ph_float = self.hurdle.predict(X)
        pt = self.tweedie.predict(X)
        wh, wt = self.weights
        combined = wh * ph + wt * pt
        # combined_float uses bias-corrected float mixed with tweedie, used before global_scale rounding
        combined_float = wh * ph_float + wt * pt
        yf = np.clip(np.round(combined), 0, None).astype(int)
        return yf, {"p_cal": pc, "pred_hurdle": ph,
                     "pred_tweedie": pt, "combined_raw": combined,
                     "combined_float": combined_float}

    def feature_importance(self, features):
        fi = self.hurdle.feature_importance(features)
        tg = self.tweedie.model.feature_importance(importance_type="gain")
        tg = tg / max(tg.sum(),1) * 100
        fi["tweedie"] = tg
        fi["combined"] = fi["cls"]*0.15 + fi["reg"]*0.35 + fi["tweedie"]*0.50
        return fi.sort_values("combined", ascending=False).reset_index(drop=True)


# ============================================================================
# Evaluation
# ============================================================================
def cmetrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    r = yp - yt; ae = np.abs(r); total = max(np.sum(yt), 1)
    return {"mae":np.mean(ae), "rmse":np.sqrt(np.mean(r**2)),
            "bias":np.mean(r), "wmape":np.sum(ae)/total,
            "hit0":np.mean(ae==0), "hit1":np.mean(ae<=1), "hit2":np.mean(ae<=2),
            "n":len(yt)}


def evaluate(model, df_test, features, v1p=None, v2p=None, v3p=None):
    print("=" * 70)
    print("V4 Evaluation")
    print("=" * 70)

    X = df_test[features]
    yt = df_test[TARGET].values
    yp, det = model.predict(X)
    m = cmetrics(yt, yp)

    print(f"\n  MAE={m['mae']:.3f}  Bias={m['bias']:+.3f}  "
          f"+/-1={m['hit1']:.1%}  +/-2={m['hit2']:.1%}")

    # Regression to mean
    print(f"\n  -- Regression to Mean --")
    for lo,hi in [(0,0),(1,2),(3,5),(6,10),(11,20),(21,999)]:
        mask = (yt>=lo)&(yt<=hi)
        if mask.sum()<5: continue
        tm = yt[mask].mean()
        pm = yp[mask].mean()
        ratio = pm / max(tm, 0.01)
        s = " OK" if 0.85<=ratio<=1.15 else " WARN" if 0.7<=ratio<=1.3 else " BAD"
        print(f"    y={lo}-{hi:>3d}: actual={tm:.1f} pred={pm:.1f} ratio={ratio:.2f} (ideal=1.00){s}")

    # False negatives
    fn = ((yt>0)&(yp==0)).sum()
    fn_m = yt[(yt>0)&(yp==0)].mean() if fn>0 else 0
    print(f"\n  FN: {fn} rows (avg actual={fn_m:.1f})")

    # By demand level
    print(f"\n  -- By Demand Level --")
    for lo,hi,lab in [(0,0,"y=0"),(1,2,"y=1-2"),(3,5,"y=3-5"),(6,10,"y=6-10"),(11,999,"y=11+")]:
        mask = (yt>=lo)&(yt<=hi)
        if mask.sum()<5: continue
        bm = cmetrics(yt[mask], yp[mask])
        print(f"    {lab:10s}  MAE={bm['mae']:.2f}  Bias={bm['bias']:+.2f}  +/-1={bm['hit1']:.0%}")

    # By week
    print(f"\n  -- By Week --")
    edf = df_test[ID_COLS].copy()
    edf["y_true"] = yt; edf["y_pred"] = yp
    edf["error"] = yp-yt; edf["abs_error"] = np.abs(edf["error"])
    edf["p_calibrated"] = det["p_cal"]
    for wk in sorted(df_test["year_week"].unique()):
        s = edf[edf["year_week"]==wk]
        wm = cmetrics(s["y_true"].values, s["y_pred"].values)
        print(f"    {wk}: MAE={wm['mae']:.3f} Bias={wm['bias']:+.3f}")

    # Multi-version comparison
    print(f"\n  {'='*60}")
    print(f"  V1 / V2 / V3 / V4")
    print(f"  {'='*60}")
    v4wks = set(edf["year_week"].unique())
    comp = {"V4": m}
    for nm, pp in [("V1",v1p),("V2",v2p),("V3",v3p)]:
        if pp and Path(pp).exists():
            prev = pd.read_csv(pp)
            prev = prev[prev["year_week"].isin(v4wks)]
            if len(prev)>0:
                comp[nm] = cmetrics(prev["y_true"].values, prev["y_pred"].values)

    vers = [v for v in ["V1","V2","V3","V4"] if v in comp]
    print(f"\n  {'':12s}" + "".join(f"  {v:>8s}" for v in vers))
    print(f"  {'-'*12}" + "  --------" * len(vers))
    for metric, fmt in [("mae", ".3f"), ("bias", ".3f"), ("hit1", ".1%"), ("hit2", ".1%")]:
        row = f"  {metric:12s}"
        for v in vers:
            row += f"  {comp[v][metric]:>8{fmt}}"
        print(row)

    # Regression-to-mean comparison
    print(f"\n  Regression-to-mean comparison:")
    for lo,hi in [(0,0),(1,2),(3,5),(6,10),(11,999)]:
        row = f"    y={lo}-{hi:>3d}:"
        for nm, pp in [("V1",v1p),("V2",v2p),("V3",v3p)]:
            if pp and Path(pp).exists():
                prev = pd.read_csv(pp)
                prev = prev[prev["year_week"].isin(v4wks)]
                pm = prev[(prev["y_true"]>=lo)&(prev["y_true"]<=hi)]
                if len(pm)>=5:
                    r = pm["y_pred"].mean() / max(pm["y_true"].mean(), 0.01)
                    row += f"  {nm}={r:.2f}"
        m4 = edf[(edf["y_true"]>=lo)&(edf["y_true"]<=hi)]
        if len(m4)>=5:
            r4 = m4["y_pred"].mean() / max(m4["y_true"].mean(), 0.01)
            row += f"  V4={r4:.2f}"
        print(row)
    print()
    return edf, m


# ============================================================================
# Out-of-sample validation (truncated week predictions vs actual totals)
# ============================================================================
def evaluate_holdout(model, df_holdout, features, actual_totals=None):
    """
    Use the trained model to predict truncated weeks (out-of-sample) and compare against actual totals.

    actual_totals: dict, e.g. {"2026-W09": 18250}
    """
    if df_holdout is None or df_holdout.empty:
        print("  (No truncated week data, skipping out-of-sample validation)")
        return df_holdout

    print("=" * 70)
    print("Holdout Prediction (Out-of-Sample Validation)")
    print("=" * 70)

    av = [c for c in features if c in df_holdout.columns]
    yp, _ = model.predict(df_holdout[av])
    df_out = df_holdout.copy()
    df_out["y_pred"] = yp

    for yw, gdf in df_out.groupby("year_week"):
        pred_total = gdf["y_pred"].sum()
        actual = (actual_totals or {}).get(str(yw))
        if actual:
            ratio = pred_total / actual
            diff  = pred_total - actual
            print(f"  {yw}: predicted={pred_total:>8,.0f}  actual={actual:>8,}  "
                  f"ratio={ratio:.2f}  diff={diff:+,.0f}")
        else:
            print(f"  {yw}: predicted={pred_total:>8,.0f}  (no actual provided)")

    out_path = OUTPUT_DIR / "holdout_predictions_v7.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}\n")
    return df_out


# ============================================================================
# Output views
# ============================================================================
def gen_views(model, df_full, features, global_scale: float = 1.0):
    """
    global_scale: global multiplier to compensate for systematic model underestimation.
      Derivation: W09 holdout measured predicted=11211 vs actual=18250 → 18250/11211=1.627
      Conservative value of 1.50 used to prevent overcorrection; update as more actual data accumulates.

    V7.2 fix: global_scale now acts on float predictions (before rounding),
    not on integer predictions (after rounding). This "revives" many stores
    that were previously rounded to 0, significantly increasing store count in the driver view.
    Example: combined_float=0.4, scale=1.5 → 0.60 → rounds to 1 (previously 0×1.5=0)
    """
    print("=" * 70)
    print("Generate Views")
    print("=" * 70)
    lw  = df_full["year_week"].max()
    lat = df_full[df_full["year_week"] == lw].copy()
    av  = [c for c in features if c in lat.columns]

    # Get raw float predictions (combined_float = float hurdle + tweedie, pre-rounding)
    _, det = model.predict(lat[av])
    yp_float = det.get("combined_float", det["combined_raw"])

    # Apply global_scale at float level, then round → more stores included in views
    if global_scale != 1.0:
        yp_float = yp_float * global_scale
        print(f"  [global_scale={global_scale:.3f} applied to float predictions]")
    yp = np.clip(np.round(yp_float), 0, None).astype(int)

    lat["pw"] = yp
    lat["pl"] = np.clip(np.floor(yp * 0.85).astype(int), 0, None)
    lat["pu"] = np.ceil(yp * 1.15).astype(int) + 1
    print(f"  week={lw}  total={yp.sum():,.0f}  zeros={int((yp==0).sum())}")

    # Global fallback ratios (used when store pattern is unavailable)
    DR_GLOBAL = {"Sunday":0.30,"Monday":0.10,"Tuesday":0.22,"Wednesday":0.08,"Thursday":0.30}
    WEEKDAY_ORDER = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,
                     "Thursday":4,"Friday":5,"Saturday":6}

    # Load per-store delivery patterns (generated by feature.py)
    pattern_path = Path(__file__).parent / "store_delivery_pattern.csv"
    store_day_ratios = {}  # {store_name: {weekday: ratio}}
    if pattern_path.exists():
        pat = pd.read_csv(pattern_path)
        for naam, grp in pat.groupby("namn"):
            delivery_days = grp[grp["is_delivery_day"] == True].copy()
            if len(delivery_days) == 0:
                # Store has no clear pattern → fall back to global
                continue
            # V7.3 Change: Always split equally among the identified delivery days,
            # ignoring the historical demand ratio.
            n = len(delivery_days)
            store_day_ratios[naam] = {row["weekday"]: 1.0 / n for _, row in delivery_days.iterrows()}
        n_with_pattern = len(store_day_ratios)
        print(f"  [Delivery patterns loaded: {n_with_pattern} stores with custom schedules]")
    else:
        print("  [store_delivery_pattern.csv not found — using global fixed ratios for all stores]")

    rows = []
    for _, r in lat.iterrows():
        wp, wl, wu = r["pw"], r["pl"], r["pu"]

        # Use store-specific delivery days. If no pattern exists, skip the store.
        store_dr = store_day_ratios.get(r["namn"], None)
        if store_dr is None:
            continue

        days = list(store_dr.keys())
        rem, rl, ru = wp, wl, wu
        for i, (d, ratio) in enumerate(store_dr.items()):
            if i == len(days) - 1:
                dp, dl, dh = rem, rl, ru
            else:
                dp = int(round(wp * ratio))
                dl = int(round(wl * ratio))
                dh = int(round(wu * ratio))
                rem -= dp; rl -= dl; ru -= dh
            day_order = WEEKDAY_ORDER.get(d, i)
            rows.append({
                "namn": r["namn"], "ort": r["ort"],
                "typ": r["typ"],   "sort": r["sort"],
                "product_id": r["product_id"],
                "year_week": r["year_week"], "day": d, "day_order": day_order,
                "predicted": max(0, dp),
                "pred_lower": max(0, dl),
                "pred_upper": max(0, dh),
            })
    daily = pd.DataFrame(rows)

    # ── Kitchen view (aggregated by product, unchanged) ───────────────────
    do_map = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
    kit = (daily.groupby(["day","typ","sort","product_id"], as_index=False)
           .agg(total=("predicted","sum"), lower=("pred_lower","sum"),
                upper=("pred_upper","sum"), stores=("namn","nunique")))
    kit["do"] = kit["day"].map(do_map)
    kit = kit.sort_values(["do","total"], ascending=[True,False])
    kit["interval"] = kit["lower"].astype(str) + "~" + kit["upper"].astype(str)
    ko = kit[["day","typ","sort","total","interval","stores"]].rename(
        columns={"day":"Day","typ":"Type","sort":"Product",
                 "total":"Qty","interval":"Range","stores":"Stores"})

    # ── Driver view (store-level) ─────────────────────────────────────────
    # Keep only rows with positive predictions
    daily_pos = daily[daily["predicted"] > 0].copy()

    # Build product detail string per (route, store, weekday)
    def build_detail(grp):
        parts = sorted([
            f"{row['typ']}/{row['sort']}: {int(row['predicted'])}"
            for _, row in grp.iterrows()
        ])
        return " | ".join(parts)

    detail_map = (
        daily_pos.groupby(["ort","namn","day"])
        .apply(build_detail, include_groups=False)
        .rename("product_detail")
        .reset_index()
    )

    # Store-level aggregation
    drv = (daily_pos.groupby(["ort","namn","day","day_order"], as_index=False)
           .agg(total_items  =("predicted",    "sum"),
                lower_total  =("pred_lower",   "sum"),
                upper_total  =("pred_upper",   "sum"),
                n_products   =("product_id",   "nunique")))
    drv = drv.merge(detail_map, on=["ort","namn","day"], how="left")
    drv["interval"] = drv["lower_total"].astype(str) + "~" + drv["upper_total"].astype(str)
    drv = drv.sort_values(["ort","day_order","namn"])
    dvo = drv[["ort","namn","day","total_items","interval","n_products","product_detail"]].rename(
        columns={"ort":"Route","namn":"Store","day":"Day",
                 "total_items":"Total_Qty","interval":"Range",
                 "n_products":"N_Products","product_detail":"Products"})

    print(f"  Kitchen view: {len(ko)} rows (weekday x product)")
    print(f"  Driver view:  {len(dvo)} rows (route x store x weekday)\n")
    return ko, dvo


# ============================================================================
# Diagnostics plot
# ============================================================================
def plot_v4(edf, fi, v1p=None, v2p=None, v3p=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed"); return

    print("=" * 70)
    print("V4 Diagnostics Plot")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Eataway V7 — Calibrated Hurdle + Bias Correction + Driver View", fontsize=14, y=0.98)

    # 1. Pred vs Actual
    ax = axes[0,0]
    s = edf.sample(min(5000,len(edf)), random_state=SEED)
    ax.scatter(s["y_true"], s["y_pred"], alpha=0.15, s=10, c="#2196F3")
    mx = max(s["y_true"].max(), s["y_pred"].max())
    ax.plot([0,mx],[0,mx],"r--",lw=1)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs Actual (V4)")

    # 2. Error distribution
    ax = axes[0,1]
    ax.hist(edf["error"], bins=np.arange(-10.5,11.5,1), color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--")
    ax.set_xlabel("Error"); ax.set_title(f"Error Distribution (mean={edf['error'].mean():.2f})")
    ax.set_xlim(-10,10)

    # 3. Regression-to-mean comparison
    ax = axes[0,2]
    labels = ["0","1-2","3-5","6-10","11+"]
    cuts = [-0.5,0.5,2.5,5.5,10.5,200]
    v4wks = set(edf["year_week"].unique())
    x = np.arange(len(labels)); width=0.18
    colors = {"V1":"#FF5722","V2":"#2196F3","V3":"#FF9800","V4":"#9C27B0"}

    for i,(nm,pp) in enumerate([("V1",v1p),("V2",v2p),("V3",v3p)]):
        if pp and Path(pp).exists():
            prev = pd.read_csv(pp)
            prev = prev[prev["year_week"].isin(v4wks)]
            if len(prev)>0:
                prev["db"] = pd.cut(prev["y_true"], bins=cuts, labels=labels)
                ratios = []
                for lb in labels:
                    sb = prev[prev["db"]==lb]
                    ratios.append(min(sb["y_pred"].mean()/max(sb["y_true"].mean(),0.01), 3.0) if len(sb)>=5 else np.nan)
                ax.bar(x+i*width, ratios, width, label=nm, color=colors[nm], alpha=0.6)

    edf["db"] = pd.cut(edf["y_true"], bins=cuts, labels=labels)
    r4 = []
    for lb in labels:
        sb = edf[edf["db"]==lb]
        r4.append(min(sb["y_pred"].mean()/max(sb["y_true"].mean(),0.01), 3.0) if len(sb)>=5 else np.nan)
    ax.bar(x+3*width, r4, width, label="V4", color=colors["V4"], alpha=0.9)
    ax.axhline(1.0,color="red",ls="--",lw=1)
    ax.set_xticks(x+1.5*width); ax.set_xticklabels(labels)
    ax.set_ylabel("Pred/Actual Ratio"); ax.set_title("Regression-to-Mean: V1->V4")
    ax.legend(fontsize=8)

    # 4. Feature importance
    ax = axes[1,0]
    top = fi.head(15).iloc[::-1]
    ax.barh(top["feature"], top["combined"], color="#FF9800", edgecolor="white")
    ax.set_xlabel("Importance (%)"); ax.set_title("Top 15 Features (V4)")

    # 5. MAE by level
    ax = axes[1,1]
    mae4 = edf.groupby("db", observed=True)["abs_error"].mean()
    ax.bar(x, mae4.reindex(labels).values, color="#9C27B0", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Demand Level"); ax.set_ylabel("MAE"); ax.set_title("MAE by Level (V4)")

    # 6. Bias by week
    ax = axes[1,2]
    wb = edf.groupby("year_week")["error"].mean()
    ax.bar(range(len(wb)), wb.values, color="#00BCD4", edgecolor="white", alpha=0.8)
    ax.axhline(0,color="red",ls="--")
    ax.set_xticks(range(len(wb))); ax.set_xticklabels(wb.index, rotation=45, fontsize=8)
    ax.set_ylabel("Bias"); ax.set_title("Bias by Week (V4)")

    plt.tight_layout()
    p = OUTPUT_DIR / "diagnostics_v7.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}\n")


# ============================================================================
# Save
# ============================================================================
def save_all(model, edf, metrics, fi, kit, drv):
    print("=" * 70)
    print("Save")
    print("=" * 70)

    model.hurdle.cls_model.save_model(str(OUTPUT_DIR / "model_cls.txt"))
    model.hurdle.reg_model.save_model(str(OUTPUT_DIR / "model_reg.txt"))
    model.tweedie.model.save_model(str(OUTPUT_DIR / "model_tweedie.txt"))

    import pickle
    with open(OUTPUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(model.hurdle.calibrator, f)

    edf.to_csv(OUTPUT_DIR / "evaluation_v7.csv", index=False)
    fi.to_csv(OUTPUT_DIR / "feature_importance_v7.csv", index=False)
    kit.to_csv(OUTPUT_DIR / "kitchen_view_v7.csv", index=False, encoding="utf-8-sig")
    drv.to_csv(OUTPUT_DIR / "driver_view_v7.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics_v7.csv", index=False)

    config = {
        "hurdle_threshold": model.hurdle.threshold,
        "bias_factors": {f"{lo}-{hi}": float(f) for (lo,hi),f in model.hurdle.bias_factors.items()},
        "ensemble_weights": {"hurdle": model.weights[0], "tweedie": model.weights[1]},
        "n_features": len(model.features) if model.features else 0,
        "has_weather": any("temp" in f for f in (model.features or [])),
    }
    with open(OUTPUT_DIR / "config_v4.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("  All saved\n")


# ============================================================================
# main
# ============================================================================
def main():
    print("=" * 70)
    print("  EATAWAY V7")
    print("  Calibrated Hurdle + Fixed Bias Correction + Store-Level Driver View")
    print("=" * 70 + "\n")

    df, features, df_holdout = load_and_prepare()
    dtr, dva, dte = time_split(df)

    model = V4Ensemble()
    model.fit(dtr, dva, features)

    v1p = None   
    v2p = None
    v3p = None

    edf, metrics = evaluate(model, dte, features, v1p, v2p, v3p)

    fi = model.feature_importance(features)
    print("=" * 70)
    print("Feature Importance Top 20")
    print("=" * 70)
    for i, row in fi.head(20).iterrows():
        bar = "#" * max(1, int(row["combined"]))
        print(f"  {i+1:>2d}. {row['feature']:30s}  {row['combined']:5.1f}%  {bar}")
    print()

    # ── Out-of-sample validation ─────────────────────────────────────────
    KNOWN_ACTUALS = {"2026-W09": 18250}
    evaluate_holdout(model, df_holdout, features, KNOWN_ACTUALS)

    # ── "Clean" prediction using the last complete training week ──────────
    # Truncated weeks (W08/W09/W10) have lag features contaminated by the previous truncated week's incomplete data:
    #   W09 lag_1w = W08_truncated(11,412) instead of W08_actual(~14,000+)
    # So predicting directly from holdout rows is unreliable.
    # Instead use the last complete week (W07) from the training set: features are 100% clean (lag=W06 complete data)
    last_clean_week = df["year_week"].max()
    clean_rows = df[df["year_week"] == last_clean_week]
    av = [c for c in features if c in clean_rows.columns]
    yp_clean, _ = model.predict(clean_rows[av])
    clean_total = yp_clean.sum()
    print("=" * 70)
    print("Clean Proxy Prediction (last complete training week)")
    print("=" * 70)
    print(f"  base_week={last_clean_week}  predicted_total={clean_total:,.0f}")
    print(f"  (W09 actual=18,250  proxy_ratio={clean_total/18250:.2f})")
    print()

    # ── Global scale factor ───────────────────────────────────────────────
    # Auto-computed from test set + seasonal demand compensation
    total_actual    = edf["y_true"].sum()
    total_predicted = edf["y_pred"].sum()
    auto_scale = total_actual / max(total_predicted, 1.0)
    # Lower bound 1.30: compensate for systematic underestimation during high-demand periods (spring, etc.)
    # Upper bound 1.60: prevent overcorrection
    # Once more years of data are available, the model will learn seasonality and the lower bound can be reduced to 1.0
    VIEW_GLOBAL_SCALE = float(np.clip(auto_scale, 1.10, 1.50))
    print(f"  Auto global_scale = {total_actual:.0f} / {total_predicted:.0f} = {auto_scale:.3f} → clipped={VIEW_GLOBAL_SCALE:.3f}\n")

    # ── Generate output views (using complete training week — clean features) ─
    kit, drv = gen_views(model, df, features, global_scale=VIEW_GLOBAL_SCALE)
    plot_v4(edf, fi, v1p, v2p, v3p)
    save_all(model, edf, metrics, fi, kit, drv)

    # ── Save predictions/ with next-week label ────────────────────────────
    from datetime import date as _dt_date, timedelta as _dt_td
    _today = _dt_date.today()

    # V7.7 Change: Week starts on Sunday. Current week output normally. If Saturday, predict next week.
    if _today.weekday() == 6:  # Sunday (start of current week for Eataway)
        _target_date = _today + _dt_td(days=1)  # Shift to Monday to get correct ISO week
    elif _today.weekday() == 5:  # Saturday (run for NEXT week starting tomorrow)
        _target_date = _today + _dt_td(days=2)  # Shift to next Monday to get correct ISO week
    else:  # Monday to Friday
        _target_date = _today
    _nyear, _nweek, _ = _target_date.isocalendar()
    target_week_label = f"{_nyear}-W{_nweek:02d}"

    PRED_DIR = Path(__file__).parent / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    # Remove old prediction files so only the latest week remains
    for _old in (list(PRED_DIR.glob("*_driver.csv")) +
                 list(PRED_DIR.glob("*_kitchen.csv")) +
                 list(PRED_DIR.glob("*_summary.txt"))):
        _old.unlink()
    drv.to_csv(PRED_DIR / f"{target_week_label}_driver.csv",  index=False, encoding="utf-8-sig")
    kit.to_csv(PRED_DIR / f"{target_week_label}_kitchen.csv", index=False, encoding="utf-8-sig")
    _total = drv["Total_Qty"].sum() if "Total_Qty" in drv.columns else 0
    (PRED_DIR / f"{target_week_label}_summary.txt").write_text(
        f"Eataway Predictions — {target_week_label}\n"
        f"Generated: {_today}\n"
        f"Total: {_total:,.0f} items\n"
        f"Base week: {last_clean_week}\n",
        encoding="utf-8")
    print(f"  Predictions saved: predictions/{target_week_label}_driver.csv")

    print("=" * 70)
    print("  V4 DONE")
    print(f"  Output: {OUTPUT_DIR}")
    wh, wt = model.weights
    print(f"  Hurdle {wh:.0%} + Tweedie {wt:.0%}")
    print(f"  Threshold: {model.hurdle.threshold:.2f}")
    hw = any("temp" in f for f in features)
    print(f"  Weather: {'YES' if hw else 'NO (run eataway_weather.py first)'}")
    print("=" * 70)

    # ── Export to Google Sheet ────────────────────────────────────────────
    print("=" * 70)
    print("Export to Google Sheet")
    print("=" * 70)
    try:
        import os as _os, re as _re
        from pathlib import Path as _Path
        from datetime import date as _date, timedelta as _td
        import gspread
        from google.oauth2.service_account import Credentials

        GSHEET_ID  = "1pRX_Mjc1Y_Xt1LYSposrLK7mkDSjMxCbF-IYDzixzlQ"
        GSHEET_TAB = "Tabell"
        SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]

        # Credentials: prefer environment variable, fall back to local credentials.json
        creds_env = _os.environ.get("GOOGLE_SHEETS_CREDS", "")
        if creds_env:
            creds = Credentials.from_service_account_info(
                json.loads(creds_env), scopes=SCOPES)
        else:
            cred_file = _Path(__file__).parent / "credentials.json"
            creds = Credentials.from_service_account_file(str(cred_file), scopes=SCOPES)

        # Delivery week = next_week_label (already computed as today + 1 week or current week)
        yw = target_week_label  # e.g. "2026-W28"
        m_ = _re.match(r"(\d{4})-W(\d{2})", yw)
        if m_:
            iso_mon  = _date.fromisocalendar(int(m_.group(1)), int(m_.group(2)), 1)
            week_sun = iso_mon - _td(days=1)   # Sunday before the delivery week's Monday
            week_sat = iso_mon + _td(days=5)   # Saturday of the delivery week
        else:
            # Fallback
            week_sun = _target_date - _td(days=(_target_date.weekday() + 1) % 7)
            week_sat = week_sun + _td(days=6)

        day_map = {}
        for offset in range((week_sat - week_sun).days + 1):
            d  = week_sun + _td(days=offset)
            dn = d.strftime("%A")
            if dn not in day_map:
                day_map[dn] = d

        # Flatten: one row per product
        sheet_rows = []
        for _, row in drv.iterrows():
            dn = row.get("Day", "")
            if dn not in day_map:
                continue
            actual_date   = day_map[dn]
            products_str  = str(row.get("Products", ""))
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
                # prod_name format: "typ/sort" from build_detail()
                if "/" in prod_name:
                    typ_part, sort_part = prod_name.split("/", 1)
                else:
                    typ_part, sort_part = "", prod_name
                sheet_rows.append([str(actual_date), str(row["Route"]),
                                   str(row["Store"]), typ_part, sort_part, qty])

        sheet_rows.sort(key=lambda r: (r[0], r[1], r[2], r[4]))

        # Write to Google Sheet (clear then overwrite)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(GSHEET_ID)
        try:
            ws = sh.worksheet(GSHEET_TAB)
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title=GSHEET_TAB, rows=5000, cols=10)
        ws.clear()
        ws.update([["Datum", "Ort", "Butik", "Typ", "Produkt", "Antal"]] + sheet_rows, "A1")
        print(f"  ✓ Written {len(sheet_rows)} rows → Google Sheet [{target_week_label}]")
    except ImportError:
        print("  ✗ Export skipped: please install dependencies with pip install gspread google-auth")
    except Exception as e:
        print(f"  ✗ Google Sheet export failed: {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()