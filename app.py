import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

APP_DIR = Path(__file__).parent

# Prefer LIVE data if it exists, otherwise fall back to sample
LIVE_PATH = APP_DIR / "data" / "live_canada_macro.csv"
SAMPLE_PATH = APP_DIR / "data" / "sample_canada_macro.csv"
DATA_PATH = LIVE_PATH if LIVE_PATH.exists() else SAMPLE_PATH

st.set_page_config(page_title="Project 1 — Canada Inflation App", layout="wide")

st.title("Project 1 — Canada Inflation & Rates Dashboard")
st.caption("MVP dashboard: load data → choose indicator → chart + key stats.")

# Show which dataset is being used
if DATA_PATH == LIVE_PATH:
    st.caption("✅ Live data (Bank of Canada) loaded")
else:
    st.caption("⚠️ Sample data loaded")

# --- Load data ---
if not DATA_PATH.exists():
    st.error(f"Missing file: {DATA_PATH}. Create sample data or fetch live data.")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

numeric_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns found. Your CSV must have numeric indicators.")
    st.stop()

# --- Sidebar controls ---
st.sidebar.header("Controls")
target = st.sidebar.selectbox("Choose indicator", numeric_cols, index=0)

# --- Layout ---
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader(f"{target} over time")

    # Simple rolling forecast (trend)
    df["trend_3m"] = df[target].rolling(window=3).mean()

    fig = px.line(
        df,
        x="date",
        y=[target, "trend_3m"],
        labels={"value": target, "variable": "Series"},
        title=f"{target}: actual vs trend",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Trend = 3-month rolling average (baseline forecast)")

with right:
    st.subheader("Key stats")
    latest = float(df[target].iloc[-1])
    avg = float(df[target].mean())
    mn = float(df[target].min())
    mx = float(df[target].max())

    st.metric("Latest", f"{latest:.2f}")
    st.metric("Average", f"{avg:.2f}")
    st.metric("Min", f"{mn:.2f}")
    st.metric("Max", f"{mx:.2f}")

st.divider()
st.subheader("Data preview")
st.dataframe(df, use_container_width=True)

st.divider()
st.header("Model comparison (simple backtest)")

# Use last N points as a test set
N_TEST = st.slider("Backtest months", min_value=2, max_value=min(6, len(df) - 2), value=3)

train_df = df.iloc[:-N_TEST].copy()
test_df = df.iloc[-N_TEST:].copy()

# Baseline: naive forecast = last train value repeated
naive_pred = [train_df[target].iloc[-1]] * N_TEST
naive_mae = float((test_df[target].values - naive_pred).__abs__().mean())

st.metric("Naive MAE", f"{naive_mae:.3f}")
st.caption("This is a simple baseline backtest (not full research-grade evaluation).")

st.divider()
st.header("Forecast (R ARIMA)")

st.write("This runs **auto.arima()** in R and shows a 12-month forecast + 95% interval.")

run_r = st.button("Run ARIMA forecast (R)")

OUT_DIR = APP_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

input_csv = OUT_DIR / "input_for_r.csv"
df[["date", target]].assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(input_csv, index=False)

out_csv = OUT_DIR / f"arima_{target}.csv"
r_script = APP_DIR / "r" / "arima_forecast.R"

if run_r:
    try:
        cmd = ["Rscript", str(r_script), str(input_csv), "date", str(target), str(out_csv)]
        subprocess.check_call(cmd)
        st.success("ARIMA forecast created.")
    except FileNotFoundError:
        st.error("Rscript not found. Install R and make sure `Rscript` is in PATH.")
    except subprocess.CalledProcessError as e:
        st.error(f"R script failed: {e}")

if out_csv.exists():
    fc = pd.read_csv(out_csv)

    last_date = df["date"].max()
    future_dates = pd.date_range(
        start=(last_date + pd.offsets.MonthBegin(1)).normalize(),
        periods=12,
        freq="MS",
    )

    fc_plot = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": fc["forecast"].values,
            "lo95": fc["lo95"].values,
            "hi95": fc["hi95"].values,
        }
    )

    st.subheader(f"{target}: Actual vs ARIMA forecast (R)")

    fig2 = px.line(title=f"{target}: Actual vs ARIMA Forecast")
    fig2.add_scatter(x=df["date"], y=df[target], mode="lines", name="Actual")
    fig2.add_scatter(x=fc_plot["date"], y=fc_plot["forecast"], mode="lines", name="ARIMA forecast")
    fig2.add_scatter(x=fc_plot["date"], y=fc_plot["hi95"], mode="lines", name="95% upper", opacity=0.25)
    fig2.add_scatter(x=fc_plot["date"], y=fc_plot["lo95"], mode="lines", name="95% lower", opacity=0.25)

    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(fc, use_container_width=True)

    with open(out_csv, "rb") as f:
        st.download_button("Download ARIMA forecast CSV", f, file_name=out_csv.name)
else:
    st.info("Click **Run ARIMA forecast (R)** to generate forecasts.")

st.divider()
st.header("Forecast (Neural Net)")

st.write("Neural forecast is loaded from a CSV you generate in Colab (free).")

nn_path = APP_DIR / "outputs" / "nn_forecasts" / f"nn_{target}.csv"

if nn_path.exists():
    nn = pd.read_csv(nn_path)
    nn["date"] = pd.to_datetime(nn["date"])

    st.subheader(f"{target}: Actual vs ARIMA vs Neural Net")

    fig3 = px.line(title=f"{target}: Actual vs ARIMA vs Neural Net")
    fig3.add_scatter(x=df["date"], y=df[target], mode="lines", name="Actual")

    # ARIMA (if exists)
    arima_path = APP_DIR / "outputs" / f"arima_{target}.csv"
    if arima_path.exists():
        fc = pd.read_csv(arima_path)
        last_date = df["date"].max()
        future_dates = pd.date_range(
            start=(last_date + pd.offsets.MonthBegin(1)).normalize(),
            periods=12,
            freq="MS",
        )
        fig3.add_scatter(x=future_dates, y=fc["forecast"], mode="lines", name="ARIMA forecast")

    # Neural
    fig3.add_scatter(x=nn["date"], y=nn["forecast"], mode="lines", name="Neural forecast")

    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(nn, use_container_width=True)

    with open(nn_path, "rb") as f:
        st.download_button("Download Neural forecast CSV", f, file_name=nn_path.name)
else:
    st.info(f"No neural forecast file yet. Create it at: {nn_path}")
    st.write("Expected columns: date, forecast")
