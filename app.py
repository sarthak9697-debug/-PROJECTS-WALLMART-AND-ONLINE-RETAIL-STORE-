from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from forecast_engine import FEATURES, REQUIRED_COLUMNS, SENSIBLE_RANGES, generate_forecast, prepare_history

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SAMPLE_PATH = BASE_DIR / "sample_data" / "walmart_sample_data.csv"

st.set_page_config(page_title="Walmart Sales Insights", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
.hero {padding: 1.25rem 1.5rem; border: 1px solid rgba(120,120,120,.24); border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .78;}
.small-note {font-size: .88rem; opacity: .8;}
</style>
<div class="hero">
  <h1>Walmart Sales Insights & 12-Week Forecast</h1>
  <p>Store-level performance, weekly sales trends, and recursive Random Forest forecasts.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl was not found beside app.py. Add the trained model artifact to the project.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_sample():
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError("Bundled sample_data/walmart_sample_data.csv was not found.")
    return pd.read_csv(SAMPLE_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

st.sidebar.header("Test the model")
source = st.sidebar.radio("Input data", ["Built-in sample data", "Upload my CSV"], index=0)
if source == "Upload my CSV":
    upload = st.sidebar.file_uploader("Upload weekly store-sales CSV", type=["csv"])
    if upload is None:
        st.info("Upload a CSV to begin, or select Built-in sample data in the sidebar.")
        st.stop()
    raw_data = pd.read_csv(upload)
    source_label = f"Uploaded file: {upload.name}"
else:
    raw_data = load_sample()
    source_label = "Built-in synthetic sample data"

horizon = st.sidebar.slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=12)

try:
    history, validation_warnings = prepare_history(raw_data)
    forecast, forecast_warnings = generate_forecast(model, history, horizon_weeks=horizon)
except Exception as exc:
    st.error(f"Input validation / forecast error: {exc}")
    st.markdown("Expected columns: `" + "`, `".join(REQUIRED_COLUMNS) + "`.")
    st.stop()

all_warnings = validation_warnings + forecast_warnings
st.caption(f"Data source: {source_label} · {len(history):,} valid observations · {history['Store'].nunique():,} stores")
st.info("Demo artifact notice: the included model.pkl was trained on the bundled synthetic sample data because the original Walmart DataSet.csv was not included with the notebook upload. Run the regenerated notebook with your genuine dataset to create a project-specific model.pkl before production use.")

if all_warnings:
    with st.expander(f"Data checks and guidance ({len(all_warnings)} notice(s))", expanded=False):
        for message in all_warnings:
            st.warning(message)

# High-level KPIs
latest_by_store = history.sort_values("Date").groupby("Store").tail(1)
metric_cols = st.columns(4)
metric_cols[0].metric("Historical sales", f"${history['Weekly_Sales'].sum():,.0f}")
metric_cols[1].metric("Average weekly sales", f"${history['Weekly_Sales'].mean():,.0f}")
metric_cols[2].metric("Stores covered", f"{history['Store'].nunique():,}")
metric_cols[3].metric("Forecast rows", f"{len(forecast):,}")

st.divider()
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Weekly sales trend")
    trend = history.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    fig = px.line(trend, x="Date", y="Weekly_Sales", markers=True,
                  labels={"Weekly_Sales": "Weekly sales ($)", "Date": "Week"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Store performance")
    store_perf = history.groupby("Store", as_index=False)["Weekly_Sales"].sum().sort_values("Weekly_Sales", ascending=False).head(10)
    fig2 = px.bar(store_perf, x="Store", y="Weekly_Sales", color="Weekly_Sales",
                  labels={"Weekly_Sales": "Total sales ($)"}, color_continuous_scale="Blues")
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Explore a store forecast")
store_ids = sorted(forecast["Store"].unique().tolist())
selected_store = st.selectbox("Select store", store_ids, index=0)
selected_forecast = forecast[forecast["Store"] == selected_store].sort_values("Date")
selected_history = history[history["Store"] == selected_store].sort_values("Date").tail(12)
forecast_total = selected_forecast["Forecast_Sales"].sum()
forecast_avg = selected_forecast["Forecast_Sales"].mean()
next_metrics = st.columns(3)
next_metrics[0].metric("Forecast total", f"${forecast_total:,.0f}")
next_metrics[1].metric("Average forecast / week", f"${forecast_avg:,.0f}")
next_metrics[2].metric("Last observed week", pd.to_datetime(selected_history["Date"].max()).strftime("%d %b %Y"))
plot_hist = selected_history[["Date", "Weekly_Sales"]].rename(columns={"Weekly_Sales": "Sales"})
plot_hist["Type"] = "Historical"
plot_fc = selected_forecast[["Date", "Forecast_Sales"]].rename(columns={"Forecast_Sales": "Sales"})
plot_fc["Type"] = "Forecast"
combined = pd.concat([plot_hist, plot_fc], ignore_index=True)
fig3 = px.line(combined, x="Date", y="Sales", color="Type", markers=True,
               labels={"Sales": "Weekly sales ($)", "Date": "Week"})
fig3.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig3, use_container_width=True)

with st.expander("View forecast table", expanded=True):
    show_forecast = selected_forecast.copy()
    show_forecast["Date"] = pd.to_datetime(show_forecast["Date"]).dt.strftime("%Y-%m-%d")
    show_forecast["Forecast_Sales"] = show_forecast["Forecast_Sales"].round(2)
    st.dataframe(show_forecast, use_container_width=True, hide_index=True)

csv_bytes = forecast.to_csv(index=False).encode("utf-8")
st.download_button("Download all-store forecast CSV", data=csv_bytes,
                   file_name="walmart_next_12_weeks_forecast.csv", mime="text/csv")

with st.expander("Input data guide and example ranges"):
    guide = pd.DataFrame([
        ["Store", "Whole-number store ID", "1–45", "1"],
        ["Date", "Weekly observation date", "Valid date; ideally 7-day spacing", "2012-12-07"],
        ["Weekly_Sales", "Observed sales for the week", "$100,000–$3,000,000 typical guide", "1,250,000"],
        ["Holiday_Flag", "Holiday-week indicator", "0 or 1", "0"],
        ["Temperature", "Temperature (Fahrenheit)", "-20 to 120", "56.4"],
        ["Fuel_Price", "Fuel price", "0.5 to 10", "3.45"],
        ["CPI", "Consumer Price Index", "50 to 400", "211.2"],
        ["Unemployment", "Unemployment rate (%)", "0 to 30", "7.8"],
    ], columns=["Column", "Description", "Usual guide range", "Example"])
    st.dataframe(guide, use_container_width=True, hide_index=True)
    st.caption("Ranges are validation guidance, not hard filters (except non-negative sales and Holiday_Flag 0/1). Every store needs at least four valid weekly observations.")
    st.download_button("Download built-in sample input CSV", data=load_sample().to_csv(index=False).encode("utf-8"),
                       file_name="walmart_sample_data.csv", mime="text/csv")

st.caption("Model: RandomForestRegressor · 300 trees · 13 engineered features · non-negative recursive forecasts")
