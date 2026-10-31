from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature",
    "Fuel_Price", "CPI", "Unemployment"
]
FEATURES = [
    "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI",
    "Unemployment", "Year", "Month", "Week", "Lag_1", "Lag_2",
    "Lag_4", "Rolling_4"
]
SENSIBLE_RANGES = {
    "Store": (1, 45),
    "Weekly_Sales": (100_000, 3_000_000),
    "Holiday_Flag": (0, 1),
    "Temperature": (-20, 120),
    "Fuel_Price": (0.5, 10),
    "CPI": (50, 400),
    "Unemployment": (0, 30),
}


def prepare_history(data: pd.DataFrame | list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    """Validate required fields and return a clean, chronologically sorted history."""
    df = pd.DataFrame(data).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("Date",)]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped = before - len(df)
    if df.empty:
        raise ValueError("No valid rows remain. Check dates and numeric input values.")

    if (df["Weekly_Sales"] < 0).any():
        raise ValueError("Weekly_Sales must be zero or greater.")
    if not df["Holiday_Flag"].isin([0, 1]).all():
        raise ValueError("Holiday_Flag must contain only 0 or 1.")
    if (df["Store"] % 1 != 0).any():
        raise ValueError("Store values must be whole-number IDs.")
    df["Store"] = df["Store"].astype(int)
    df["Holiday_Flag"] = df["Holiday_Flag"].astype(int)

    warnings: list[str] = []
    if dropped:
        warnings.append(f"Removed {dropped:,} row(s) with missing or invalid required values.")
    duplicate_pairs = int(df.duplicated(subset=["Store", "Date"]).sum())
    if duplicate_pairs:
        warnings.append(f"Found {duplicate_pairs:,} repeated Store/Date rows; keeping the last row for each week.")
        df = df.drop_duplicates(subset=["Store", "Date"], keep="last")

    for col, (low, high) in SENSIBLE_RANGES.items():
        out_count = int((~df[col].between(low, high)).sum())
        if out_count:
            warnings.append(f"{out_count:,} {col} value(s) fall outside the usual guide range {low:g}–{high:g}; values are retained.")

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    counts = df.groupby("Store").size()
    short = counts[counts < 4]
    if not short.empty:
        raise ValueError(
            "Each store needs at least 4 valid weekly observations to create lag features. "
            "Insufficient history for store(s): " + ", ".join(map(str, short.index.tolist()))
        )
    # A gap is not fatal, but it is useful to tell the user what the model assumes.
    irregular = []
    for store, grp in df.groupby("Store"):
        dates = grp["Date"].sort_values()
        if len(dates) > 1 and not dates.diff().dropna().eq(pd.Timedelta(days=7)).all():
            irregular.append(int(store))
    if irregular:
        warnings.append("Some store histories are not strictly 7 days apart (stores: " + ", ".join(map(str, irregular[:10])) + ("…" if len(irregular) > 10 else "") + "). Forecast dates still advance in 7-day steps.")
    return df, warnings


def generate_forecast(model, data: pd.DataFrame | list[dict[str, Any]], horizon_weeks: int = 12) -> tuple[pd.DataFrame, list[str]]:
    """Generate recursive store-level forecasts using the notebook's 13 features."""
    if not isinstance(horizon_weeks, int) or not 1 <= horizon_weeks <= 12:
        raise ValueError("horizon_weeks must be an integer from 1 to 12.")
    history, warnings = prepare_history(data)
    model_features = list(getattr(model, "feature_names_in_", FEATURES))
    missing_features = [f for f in FEATURES if f not in model_features]
    if missing_features:
        raise ValueError("The loaded model does not contain the expected feature set.")

    forecast_rows: list[dict[str, Any]] = []
    for store, store_history in history.groupby("Store", sort=True):
        store_history = store_history.sort_values("Date").copy()
        sales_history = store_history["Weekly_Sales"].astype(float).tolist()
        latest = store_history.iloc[-1]
        last_date = pd.Timestamp(latest["Date"])
        for _ in range(horizon_weeks):
            next_date = last_date + pd.Timedelta(weeks=1)
            row = {
                "Store": int(store),
                # Mirrors the notebook's assumption for future weeks.
                "Holiday_Flag": 0,
                "Temperature": float(latest["Temperature"]),
                "Fuel_Price": float(latest["Fuel_Price"]),
                "CPI": float(latest["CPI"]),
                "Unemployment": float(latest["Unemployment"]),
                "Year": int(next_date.year),
                "Month": int(next_date.month),
                "Week": int(next_date.isocalendar().week),
                "Lag_1": float(sales_history[-1]),
                "Lag_2": float(sales_history[-2]),
                "Lag_4": float(sales_history[-4]),
                "Rolling_4": float(np.mean(sales_history[-4:])),
            }
            input_row = pd.DataFrame([row], columns=model_features)
            prediction = max(0.0, float(model.predict(input_row)[0]))
            forecast_rows.append({
                "Store": int(store),
                "Date": next_date,
                "Forecast_Sales": prediction,
            })
            sales_history.append(prediction)
            last_date = next_date

    return pd.DataFrame(forecast_rows), warnings
