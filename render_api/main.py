from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from forecast_engine import REQUIRED_COLUMNS, generate_forecast

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SAMPLE_PATH = BASE_DIR.parent / "sample_data" / "walmart_sample_data.csv"

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:
    MODEL = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None

app = FastAPI(
    title="Walmart Weekly Sales Forecast API",
    version="1.0.0",
    description="Generate up to 12 weeks of store-level Walmart weekly sales forecasts from recent historical observations."
)

class ForecastRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "horizon_weeks": 12,
            "records": [
                {"Store": 1, "Date": "2012-11-09", "Weekly_Sales": 1520000, "Holiday_Flag": 0, "Temperature": 55.2, "Fuel_Price": 3.45, "CPI": 211.2, "Unemployment": 7.8},
                {"Store": 1, "Date": "2012-11-16", "Weekly_Sales": 1540000, "Holiday_Flag": 0, "Temperature": 53.7, "Fuel_Price": 3.46, "CPI": 211.3, "Unemployment": 7.8},
                {"Store": 1, "Date": "2012-11-23", "Weekly_Sales": 1680000, "Holiday_Flag": 1, "Temperature": 48.9, "Fuel_Price": 3.47, "CPI": 211.4, "Unemployment": 7.7},
                {"Store": 1, "Date": "2012-11-30", "Weekly_Sales": 1560000, "Holiday_Flag": 0, "Temperature": 50.1, "Fuel_Price": 3.48, "CPI": 211.5, "Unemployment": 7.7}
            ]
        }
    })
    records: list[dict[str, Any]] = Field(min_length=4, description="Rows with the eight required Walmart weekly sales columns.")
    horizon_weeks: int = Field(default=12, ge=1, le=12, description="Forecast length from 1 to 12 weeks.")

@app.get("/", tags=["Service"])
def root():
    return {
        "service": "Walmart Weekly Sales Forecast API",
        "docs": "/docs",
        "health": "/health",
        "forecast_endpoint": "POST /forecast",
        "input_columns": REQUIRED_COLUMNS,
    }

@app.get("/health", tags=["Service"])
def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_LOAD_ERROR}")
    return {"status": "ok", "model_loaded": True, "model_type": type(MODEL).__name__}

@app.get("/sample-data", tags=["Testing"])
def sample_data(store_id: int = 1, rows: int = 8):
    """Return a compact sample history for trying POST /forecast."""
    if not SAMPLE_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample CSV is not packaged with this deployment.")
    df = pd.read_csv(SAMPLE_PATH)
    df = df[df["Store"] == store_id].sort_values("Date").tail(max(4, min(rows, 60)))
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Store {store_id} is not present in sample data.")
    return {"records": df[REQUIRED_COLUMNS].to_dict(orient="records"), "hint": "POST this records array to /forecast."}

@app.post("/forecast", tags=["Forecast"])
def forecast(request: ForecastRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {MODEL_LOAD_ERROR}")
    try:
        forecast_df, warnings = generate_forecast(
            MODEL, request.records, horizon_weeks=request.horizon_weeks
        )
        result = forecast_df.copy()
        result["Date"] = pd.to_datetime(result["Date"]).dt.strftime("%Y-%m-%d")
        result["Forecast_Sales"] = result["Forecast_Sales"].round(2)
        return {
            "forecast_horizon_weeks": request.horizon_weeks,
            "store_count": int(result["Store"].nunique()),
            "forecast_row_count": int(len(result)),
            "warnings": warnings,
            "forecasts": result.to_dict(orient="records"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Forecast failed: {exc}") from exc
