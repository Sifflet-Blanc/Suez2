import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

LEAKY_STATIC_COLS = [
    "rise_rate",
    "fall_rate",
    "autocorr",
    "annual_variab",
    "recovery_constante",
    "recession_constante",
    "average_seasonal_fluctuations",
    "parde_season",
    "bimodal",
    "richards",
]

STATIC_MIN_COLS = [
    "BSS_ID",
    "H",
    "XL93",
    "YL93",
    "formation",
    "etat",
    "nature",
    "milieu",
    "origine",
    "theme",
]

DEFAULT_CONFIG = {
    "lags": [1, 7, 30, 90],
    "rolling_windows": [7, 30, 90],
    "max_rows": None,
    "holdout_ids": None,
    "feature_mode": "ar",
    "random_state": 42,
    "model": {
        "max_depth": 8,
        "max_iter": 200,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
    },
}


class DataError(RuntimeError):
    pass


def _read_static(data_dir: Path) -> pd.DataFrame:
    static_path = data_dir / "static_attributes" / "piezo_characs.csv"
    if not static_path.exists():
        raise DataError(f"Missing static file: {static_path}")
    df = pd.read_csv(static_path, sep=";", encoding="latin-1")
    df = df.rename(columns={"\u00e9tat": "etat", "th\u00e8me": "theme"})
    # Convert start_r to numeric features when possible.
    if "start_r" in df.columns:
        dt = pd.to_datetime(df["start_r"], errors="coerce", dayfirst=True)
        df["start_r_year"] = dt.dt.year
        df["start_r_doy"] = dt.dt.dayofyear
        df = df.drop(columns=["start_r"])
    # Coerce object columns to numeric when most values look numeric.
    for col in df.columns:
        if col == "BSS_ID":
            continue
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.8:
                df[col] = converted
    return df


def _drop_high_cardinality(df: pd.DataFrame, max_unique_ratio: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols: List[str] = []
    for col in df.columns:
        if col == "BSS_ID":
            continue
        if df[col].dtype == object:
            unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
            if unique_ratio > max_unique_ratio:
                drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols


def load_static(data_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = _read_static(data_dir)
    dropped: List[str] = []
    leaky_cols = [c for c in LEAKY_STATIC_COLS if c in df.columns]
    if leaky_cols:
        df = df.drop(columns=leaky_cols)
        dropped.extend(leaky_cols)
    df, dropped_high = _drop_high_cardinality(df)
    dropped.extend(dropped_high)
    return df, dropped


def _filter_static_by_mode(df: pd.DataFrame, feature_mode: str) -> Tuple[pd.DataFrame, List[str]]:
    if feature_mode != "static_min":
        return df, []
    keep_cols = [c for c in STATIC_MIN_COLS if c in df.columns]
    if "BSS_ID" not in keep_cols:
        keep_cols.insert(0, "BSS_ID")
    dropped = [c for c in df.columns if c not in keep_cols]
    return df[keep_cols].copy(), dropped


def list_series_ids(data_dir: Path) -> List[str]:
    ts_dir = data_dir / "time_series"
    return sorted(p.stem for p in ts_dir.glob("*.csv"))


def load_series_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Unnamed: 0", ""]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "date_mesure" not in df.columns or "niveau_nappe_eau" not in df.columns:
        raise DataError(f"Missing required columns in {path.name}")
    df = df.rename(columns={"date_mesure": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["niveau_nappe_eau"] = pd.to_numeric(df["niveau_nappe_eau"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.groupby("date", as_index=False)["niveau_nappe_eau"].mean()
    df = df.sort_values("date")
    df = df.set_index("date").asfreq("D")
    return df


def build_dynamic_features(
    df: pd.DataFrame,
    lags: Iterable[int],
    windows: Iterable[int],
    use_target_features: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    if "niveau_nappe_eau" not in df.columns:
        df["niveau_nappe_eau"] = np.nan
    df["y"] = df["niveau_nappe_eau"]
    idx = df.index
    day_of_year = idx.dayofyear
    df["doy"] = day_of_year
    df["sin_doy"] = np.sin(2 * math.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * math.pi * day_of_year / 365.25)
    if use_target_features:
        df["delta_1"] = df["y"] - df["y"].shift(1)
        for lag in lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for window in windows:
            shifted = df["y"].shift(1)
            df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            df[f"roll_std_{window}"] = shifted.rolling(window).std()
    df = df.reset_index().rename(columns={"index": "date"})
    return df


def build_supervised_dataset(
    data_dir: Path,
    static_df: pd.DataFrame,
    bss_ids: Iterable[str],
    lags: Iterable[int],
    windows: Iterable[int],
    use_target_features: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[pd.DataFrame] = []
    dynamic_cols: Optional[List[str]] = None
    ts_dir = data_dir / "time_series"
    for bss_id in bss_ids:
        ts_path = ts_dir / f"{bss_id}.csv"
        if not ts_path.exists():
            continue
        ts_df = load_series_file(ts_path)
        dyn = build_dynamic_features(ts_df, lags, windows, use_target_features=use_target_features)
        if dynamic_cols is None:
            dynamic_cols = [
                c
                for c in dyn.columns
                if c not in ["date", "y", "niveau_nappe_eau"]
            ]
        dyn["BSS_ID"] = bss_id
        rows.append(dyn)
    if not rows:
        raise DataError("No time series loaded.")
    data = pd.concat(rows, ignore_index=True)
    data = data.merge(static_df, on="BSS_ID", how="left")
    if dynamic_cols is None:
        dynamic_cols = []
    return data, dynamic_cols


def split_train_test(data: pd.DataFrame, holdout_ids: Optional[List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    bss_ids = sorted(data["BSS_ID"].unique().tolist())
    if holdout_ids is None:
        holdout_ids = [bss_ids[0]] if bss_ids else []
    train_df = data[~data["BSS_ID"].isin(holdout_ids)].copy()
    test_df = data[data["BSS_ID"].isin(holdout_ids)].copy()
    return train_df, test_df, holdout_ids


def nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return float("nan")
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    penalty = (2 / alpha) * ((lower - y_true) * below + (y_true - upper) * above)
    return np.mean(width + penalty)


def _build_pipeline(num_cols: List[str], cat_cols: List[str], model_params: Dict) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    model = HistGradientBoostingRegressor(random_state=42, **model_params)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _align_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[feature_cols]


def train_model(
    data_dir: Path,
    model_dir: Path,
    config: Optional[Dict] = None,
) -> Dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model_params = {**DEFAULT_CONFIG["model"], **cfg.get("model", {})}
    feature_mode = str(cfg.get("feature_mode", "ar")).lower()
    if feature_mode not in ["ar", "static", "static_min"]:
        raise DataError(f"Unsupported feature_mode: {feature_mode}")
    use_target_features = feature_mode == "ar"

    static_df, dropped_cols = load_static(data_dir)
    static_df, dropped_mode = _filter_static_by_mode(static_df, feature_mode)
    dropped_cols.extend(dropped_mode)
    series_ids = list_series_ids(data_dir)

    data, dynamic_cols = build_supervised_dataset(
        data_dir,
        static_df,
        series_ids,
        cfg["lags"],
        cfg["rolling_windows"],
        use_target_features=use_target_features,
    )

    feature_cols = [c for c in data.columns if c not in ["date", "BSS_ID", "y", "niveau_nappe_eau"]]

    # Drop rows with missing target or dynamic features.
    data = data.dropna(subset=["y"])
    if use_target_features and dynamic_cols:
        data = data.dropna(subset=dynamic_cols)

    if cfg.get("max_rows"):
        data = data.sample(n=cfg["max_rows"], random_state=cfg["random_state"])

    train_df, test_df, holdout_ids = split_train_test(data, cfg.get("holdout_ids"))

    X_train = _align_features(train_df, feature_cols)
    y_train = train_df["y"].to_numpy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pipeline = _build_pipeline(num_cols, cat_cols, model_params)
    pipeline.fit(X_train, y_train)

    # Residual-based intervals using training data.
    y_train_pred = pipeline.predict(X_train)
    residuals = y_train - y_train_pred
    q_low, q_high = np.quantile(residuals, [0.025, 0.975])

    metrics = {}
    if len(test_df) > 0:
        X_test = _align_features(test_df, feature_cols)
        y_test = test_df["y"].to_numpy()
        y_pred = pipeline.predict(X_test)
        lower = y_pred + q_low
        upper = y_pred + q_high
        metrics = {
            "nse": float(nash_sutcliffe(y_test, y_pred)),
            "winkler": float(winkler_score(y_test, lower, upper)),
            "coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")

    date_min = str(data["date"].min())
    date_max = str(data["date"].max())

    meta = {
        "config": cfg,
        "model_params": model_params,
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "dynamic_cols": dynamic_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dropped_static_cols": dropped_cols,
        "holdout_ids": holdout_ids,
        "residual_quantiles": {"q_low": float(q_low), "q_high": float(q_high)},
        "date_range": {"start": date_min, "end": date_max},
    }
    with (model_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if metrics:
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return {"model_dir": str(model_dir), "metrics": metrics}


def _build_date_range(meta: Dict) -> pd.DatetimeIndex:
    start = pd.to_datetime(meta["date_range"]["start"])
    end = pd.to_datetime(meta["date_range"]["end"])
    return pd.date_range(start, end, freq="D")


def _make_prediction_frame(
    bss_id: str,
    dyn: pd.DataFrame,
    feature_cols: List[str],
    pipeline: Pipeline,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    X = _align_features(dyn, feature_cols)
    preds = pipeline.predict(X)
    lower = preds + q_low
    upper = preds + q_high
    out = pd.DataFrame(
        {
            "BSS_ID": bss_id,
            "date": dyn["date"],
            "y_obs": dyn.get("y"),
            "y_pred": preds,
            "y_lower": lower,
            "y_upper": upper,
        }
    )
    return out


def compute_etiage_means(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp[tmp["date"].dt.month.isin([7, 8, 9])]
    tmp["year"] = tmp["date"].dt.year
    out = tmp.groupby(["BSS_ID", "year"])[value_col].mean().reset_index()
    out = out.rename(columns={value_col: "etiage_mean"})
    return out


def evaluate_leave_one_out(
    data_dir: Path,
    out_dir: Path,
    config: Optional[Dict] = None,
    feature_mode: Optional[str] = None,
    max_items: Optional[int] = None,
) -> Dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if feature_mode:
        cfg["feature_mode"] = feature_mode
    model_params = {**DEFAULT_CONFIG["model"], **cfg.get("model", {})}
    feature_mode = str(cfg.get("feature_mode", "ar")).lower()
    if feature_mode not in ["ar", "static", "static_min"]:
        raise DataError(f"Unsupported feature_mode: {feature_mode}")
    use_target_features = feature_mode == "ar"

    static_df, dropped_cols = load_static(data_dir)
    static_df, dropped_mode = _filter_static_by_mode(static_df, feature_mode)
    dropped_cols.extend(dropped_mode)
    series_ids = list_series_ids(data_dir)

    data, dynamic_cols = build_supervised_dataset(
        data_dir,
        static_df,
        series_ids,
        cfg["lags"],
        cfg["rolling_windows"],
        use_target_features=use_target_features,
    )

    feature_cols = [c for c in data.columns if c not in ["date", "BSS_ID", "y", "niveau_nappe_eau"]]
    data = data.dropna(subset=["y"])
    if use_target_features and dynamic_cols:
        data = data.dropna(subset=dynamic_cols)

    bss_ids = sorted(data["BSS_ID"].unique().tolist())
    if max_items:
        bss_ids = bss_ids[:max_items]

    results = []
    for holdout in bss_ids:
        train_df = data[data["BSS_ID"] != holdout].copy()
        test_df = data[data["BSS_ID"] == holdout].copy()
        if train_df.empty or test_df.empty:
            continue

        X_train = _align_features(train_df, feature_cols)
        y_train = train_df["y"].to_numpy()
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in feature_cols if c not in cat_cols]

        pipeline = _build_pipeline(num_cols, cat_cols, model_params)
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        residuals = y_train - y_train_pred
        q_low, q_high = np.quantile(residuals, [0.025, 0.975])

        X_test = _align_features(test_df, feature_cols)
        y_test = test_df["y"].to_numpy()
        y_pred = pipeline.predict(X_test)
        lower = y_pred + q_low
        upper = y_pred + q_high

        results.append(
            {
                "BSS_ID": holdout,
                "nse": float(nash_sutcliffe(y_test, y_pred)),
                "winkler": float(winkler_score(y_test, lower, upper)),
                "coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
                "n_obs": int(len(y_test)),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = out_dir / f"loo_{feature_mode}.csv"
    results_df.to_csv(results_path, index=False)

    summary = {
        "feature_mode": feature_mode,
        "stations": int(results_df.shape[0]),
        "nse_mean": float(results_df["nse"].mean()) if not results_df.empty else float("nan"),
        "nse_median": float(results_df["nse"].median()) if not results_df.empty else float("nan"),
        "coverage_mean": float(results_df["coverage"].mean()) if not results_df.empty else float("nan"),
        "winkler_mean": float(results_df["winkler"].mean()) if not results_df.empty else float("nan"),
        "dropped_static_cols": dropped_cols,
        "results_csv": str(results_path),
    }

    summary_path = out_dir / f"loo_{feature_mode}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def evaluate_group_kfold(
    data_dir: Path,
    out_dir: Path,
    config: Optional[Dict] = None,
    feature_mode: Optional[str] = None,
    n_splits: int = 5,
) -> Dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if feature_mode:
        cfg["feature_mode"] = feature_mode
    model_params = {**DEFAULT_CONFIG["model"], **cfg.get("model", {})}
    feature_mode = str(cfg.get("feature_mode", "ar")).lower()
    if feature_mode not in ["ar", "static", "static_min"]:
        raise DataError(f"Unsupported feature_mode: {feature_mode}")
    use_target_features = feature_mode == "ar"

    static_df, dropped_cols = load_static(data_dir)
    static_df, dropped_mode = _filter_static_by_mode(static_df, feature_mode)
    dropped_cols.extend(dropped_mode)
    series_ids = list_series_ids(data_dir)

    data, dynamic_cols = build_supervised_dataset(
        data_dir,
        static_df,
        series_ids,
        cfg["lags"],
        cfg["rolling_windows"],
        use_target_features=use_target_features,
    )

    feature_cols = [c for c in data.columns if c not in ["date", "BSS_ID", "y", "niveau_nappe_eau"]]
    data = data.dropna(subset=["y"])
    if use_target_features and dynamic_cols:
        data = data.dropna(subset=dynamic_cols)

    groups = data["BSS_ID"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups), start=1):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()
        if train_df.empty or test_df.empty:
            continue

        X_train = _align_features(train_df, feature_cols)
        y_train = train_df["y"].to_numpy()
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in feature_cols if c not in cat_cols]

        pipeline = _build_pipeline(num_cols, cat_cols, model_params)
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        residuals = y_train - y_train_pred
        q_low, q_high = np.quantile(residuals, [0.025, 0.975])

        X_test = _align_features(test_df, feature_cols)
        y_test = test_df["y"].to_numpy()
        y_pred = pipeline.predict(X_test)
        lower = y_pred + q_low
        upper = y_pred + q_high

        results.append(
            {
                "fold": int(fold_idx),
                "nse": float(nash_sutcliffe(y_test, y_pred)),
                "winkler": float(winkler_score(y_test, lower, upper)),
                "coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
                "n_obs": int(len(y_test)),
                "n_stations": int(test_df["BSS_ID"].nunique()),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = out_dir / f"groupkfold_{feature_mode}.csv"
    results_df.to_csv(results_path, index=False)

    summary = {
        "feature_mode": feature_mode,
        "folds": int(results_df.shape[0]),
        "nse_mean": float(results_df["nse"].mean()) if not results_df.empty else float("nan"),
        "nse_median": float(results_df["nse"].median()) if not results_df.empty else float("nan"),
        "coverage_mean": float(results_df["coverage"].mean()) if not results_df.empty else float("nan"),
        "winkler_mean": float(results_df["winkler"].mean()) if not results_df.empty else float("nan"),
        "dropped_static_cols": dropped_cols,
        "results_csv": str(results_path),
    }

    summary_path = out_dir / f"groupkfold_{feature_mode}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def evaluate_time_split(
    data_dir: Path,
    out_dir: Path,
    config: Optional[Dict] = None,
    feature_mode: Optional[str] = None,
    test_ratio: float = 0.2,
    min_obs: int = 365,
) -> Dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if feature_mode:
        cfg["feature_mode"] = feature_mode
    model_params = {**DEFAULT_CONFIG["model"], **cfg.get("model", {})}
    feature_mode = str(cfg.get("feature_mode", "ar")).lower()
    if feature_mode not in ["ar", "static", "static_min"]:
        raise DataError(f"Unsupported feature_mode: {feature_mode}")
    use_target_features = feature_mode == "ar"

    static_df, dropped_cols = load_static(data_dir)
    static_df, dropped_mode = _filter_static_by_mode(static_df, feature_mode)
    dropped_cols.extend(dropped_mode)
    series_ids = list_series_ids(data_dir)

    data, dynamic_cols = build_supervised_dataset(
        data_dir,
        static_df,
        series_ids,
        cfg["lags"],
        cfg["rolling_windows"],
        use_target_features=use_target_features,
    )

    feature_cols = [c for c in data.columns if c not in ["date", "BSS_ID", "y", "niveau_nappe_eau"]]
    data = data.dropna(subset=["y"])
    if use_target_features and dynamic_cols:
        data = data.dropna(subset=dynamic_cols)

    split_dates: Dict[str, pd.Timestamp] = {}
    for bss_id, g in data.groupby("BSS_ID"):
        g = g.sort_values("date")
        if len(g) < min_obs:
            continue
        split_idx = max(1, int(len(g) * (1 - test_ratio)))
        split_dates[bss_id] = g["date"].iloc[split_idx - 1]

    if not split_dates:
        raise DataError("No stations meet the minimum observations requirement.")

    data = data[data["BSS_ID"].isin(split_dates.keys())]
    split_series = data["BSS_ID"].map(split_dates)
    train_df = data[data["date"] <= split_series].copy()
    test_df = data[data["date"] > split_series].copy()

    if train_df.empty or test_df.empty:
        raise DataError("Time split produced empty train or test set.")

    X_train = _align_features(train_df, feature_cols)
    y_train = train_df["y"].to_numpy()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pipeline = _build_pipeline(num_cols, cat_cols, model_params)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    residuals = y_train - y_train_pred
    q_low, q_high = np.quantile(residuals, [0.025, 0.975])

    X_test = _align_features(test_df, feature_cols)
    y_test = test_df["y"].to_numpy()
    y_pred = pipeline.predict(X_test)
    lower = y_pred + q_low
    upper = y_pred + q_high

    global_metrics = {
        "nse": float(nash_sutcliffe(y_test, y_pred)),
        "winkler": float(winkler_score(y_test, lower, upper)),
        "coverage": float(np.mean((y_test >= lower) & (y_test <= upper))),
        "n_obs": int(len(y_test)),
        "n_stations": int(test_df["BSS_ID"].nunique()),
    }

    per_station = []
    for bss_id, g in test_df.groupby("BSS_ID"):
        y_true = g["y"].to_numpy()
        y_hat = pipeline.predict(_align_features(g, feature_cols))
        denom = np.sum((y_true - np.mean(y_true)) ** 2)
        if denom == 0:
            continue
        nse = 1 - np.sum((y_true - y_hat) ** 2) / denom
        per_station.append({"BSS_ID": bss_id, "nse": float(nse), "n_obs": int(len(y_true))})

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(per_station)
    results_path = out_dir / f"timesplit_{feature_mode}.csv"
    results_df.to_csv(results_path, index=False)

    summary = {
        "feature_mode": feature_mode,
        "test_ratio": float(test_ratio),
        "min_obs": int(min_obs),
        "global": global_metrics,
        "nse_mean": float(results_df["nse"].mean()) if not results_df.empty else float("nan"),
        "nse_median": float(results_df["nse"].median()) if not results_df.empty else float("nan"),
        "dropped_static_cols": dropped_cols,
        "results_csv": str(results_path),
    }

    summary_path = out_dir / f"timesplit_{feature_mode}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def reconstruct(
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
    bss_ids: Optional[List[str]] = None,
) -> Dict:
    with (model_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pipeline = joblib.load(model_dir / "model.joblib")

    feature_mode = str(meta.get("feature_mode", meta.get("config", {}).get("feature_mode", "ar"))).lower()
    use_target_features = feature_mode == "ar"

    static_df, _ = load_static(data_dir)
    static_df, _ = _filter_static_by_mode(static_df, feature_mode)
    if bss_ids is None:
        bss_ids = list_series_ids(data_dir)

    date_range = _build_date_range(meta)
    feature_cols = meta["feature_cols"]
    dynamic_cols = meta.get("dynamic_cols", [])
    q_low = meta["residual_quantiles"]["q_low"]
    q_high = meta["residual_quantiles"]["q_high"]

    preds: List[pd.DataFrame] = []
    ts_dir = data_dir / "time_series"

    for bss_id in bss_ids:
        ts_path = ts_dir / f"{bss_id}.csv"
        if ts_path.exists():
            ts_df = load_series_file(ts_path)
        else:
            ts_df = pd.DataFrame(index=date_range)
        dyn = build_dynamic_features(
            ts_df,
            meta["config"]["lags"],
            meta["config"]["rolling_windows"],
            use_target_features=use_target_features,
        )
        dyn["BSS_ID"] = bss_id
        dyn = dyn.merge(static_df, on="BSS_ID", how="left")

        # Decide if we can drop rows with missing dynamic features (only if we have data).
        has_obs = dyn["y"].notna().sum() > 0
        if has_obs and use_target_features and dynamic_cols:
            dyn = dyn.dropna(subset=dynamic_cols)
        if dyn.empty:
            continue
        pred_df = _make_prediction_frame(bss_id, dyn, feature_cols, pipeline, q_low, q_high)
        preds.append(pred_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_all = pd.concat(preds, ignore_index=True)
    pred_all.to_csv(out_dir / "predictions.csv", index=False)

    etiage = compute_etiage_means(pred_all, "y_pred")
    etiage.to_csv(out_dir / "etiage_means.csv", index=False)

    return {"out_dir": str(out_dir), "rows": len(pred_all)}
