# Walmart Sales Insights & 12-Week Forecast

A Random Forest regression project that analyzes weekly store sales, surfaces store-level performance insights, and forecasts up to 12 weeks ahead. Includes a recruiter-friendly Streamlit dashboard, built-in sample CSV, and a FastAPI service ready for Render.

## Streamlit dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with bundled sample data. Upload a CSV with the eight required columns to test your own history; each store needs at least four valid weekly observations.

## Render API

Deploy the `render_api` folder as the Render service root. Use:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

After deployment, open `/docs`, call `/health`, or get a compact test input from `/sample-data?store_id=1&rows=8` and POST its `records` to `/forecast`.

## Notebook and data note

The regenerated notebook preserves the original analysis cells and adds the recursive forecast completion plus a `model.pkl` save step. `Walmart DataSet.csv` and `sample_data/walmart_sample_data.csv` in this package are **synthetic demonstration data** because the original `Walmart DataSet.csv` was not included with the notebook upload. The included `model.pkl` is trained on that synthetic sample so the apps are testable immediately. Replace the CSV with the genuine dataset and run the notebook from top to bottom to create a project-specific `model.pkl` before presenting it as a model trained on real Walmart data.

## Input columns

`Store`, `Date`, `Weekly_Sales`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, and `Unemployment`. Future holiday flags are set to 0 and the latest observed macro/weather values are carried forward, following the notebook's stated forecasting assumptions.
