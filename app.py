from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
import json

try:
    import requests
except Exception:
    requests = None

from src.data_generator import generate_synthetic_complaints
from src.genai import generate_draft_response
from src.pipeline import process_complaints


st.set_page_config(page_title="PS5 Unified Complaint Dashboard", page_icon="📊", layout="wide")

st.title("PS5: Unified Customer Complaint Communication Dashboard")
st.caption("POC flow: Multi-channel complaint intake -> NLP categorization -> sentiment/urgency -> duplicate detection -> Gen-AI draft responses")

with st.sidebar:
    st.header("POC Controls")
    data_mode = st.radio("Dataset source", ["Synthetic demo data", "Upload CSV"], index=0)
    row_count = st.slider("Number of synthetic complaints", min_value=100, max_value=1200, value=350, step=50)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    uploaded_file = None
    if data_mode == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a complaint CSV",
            type=["csv"],
            help="CSV must contain complaint_text and category_label columns. If you use a Kaggle dataset, map or preprocess it to this schema first.",
        )
    provider = st.selectbox(
        "Draft response provider",
        ["template", "openai"],
        help="OpenAI mode needs OPENAI_API_KEY. Falls back to template if unavailable.",
    )
    use_api = st.checkbox("Use FastAPI backend (live)")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    run_btn = st.button("Run Pipeline", type="primary")

if run_btn or "processed_df" not in st.session_state:
    if data_mode == "Upload CSV" and uploaded_file is not None:
        raw_df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    else:
        raw_df = generate_synthetic_complaints(n=row_count, seed=int(seed))

    required_columns = {"complaint_text", "category_label"}
    missing_columns = required_columns.difference(raw_df.columns)
    if missing_columns:
        st.error("Uploaded data is missing required columns: " + ", ".join(sorted(missing_columns)))
        st.stop()

    # If using API backend, POST data and fetch processed results
    if use_api:
        if requests is None:
            st.error("`requests` package is not installed. Install it to use API backend.")
            st.stop()

        try:
            if data_mode == "Upload CSV" and uploaded_file is not None:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                resp = requests.post(f"{backend_url.rstrip('/')}/upload-csv", files=files, timeout=20)
                resp.raise_for_status()
            else:
                # send synthetic dataframe as CSV bytes
                csv_bytes = raw_df.to_csv(index=False).encode("utf-8")
                files = {"file": ("synthetic.csv", csv_bytes, "text/csv")}
                resp = requests.post(f"{backend_url.rstrip('/')}/upload-csv", files=files, timeout=30)
                resp.raise_for_status()

            proc = requests.get(f"{backend_url.rstrip('/')}/processed", timeout=20)
            proc.raise_for_status()
            processed_json = proc.json()
            processed_df = pd.DataFrame(processed_json)

            # fetch insights if available
            try:
                ins = requests.get(f"{backend_url.rstrip('/')}/insights", timeout=10)
                ins.raise_for_status()
                insights = ins.json()
            except Exception:
                insights = None

        except Exception as e:
            st.error(f"API backend error: {e}. Falling back to local processing.")
            processed_df, _ = process_complaints(raw_df)
            insights = None
    else:
        processed_df, _ = process_complaints(raw_df)
        insights = None
    processed_df["draft_response"] = processed_df.apply(
        lambda r: generate_draft_response(
            complaint_text=r["complaint_text"],
            predicted_category=r["predicted_category"],
            urgency=r["urgency"],
            provider=provider,
        ),
        axis=1,
    )

    st.session_state["processed_df"] = processed_df

processed_df = st.session_state["processed_df"]

st.subheader("Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Complaints", f"{len(processed_df):,}")
col2.metric("High Urgency", int((processed_df["urgency"] == "High").sum()))
col3.metric("Duplicate Flagged", int(processed_df["is_duplicate"].sum()))
col4.metric("SLA Breaches", int((processed_df["sla_status"] == "Breached").sum()))
col5.metric("Negative Sentiment", int((processed_df["sentiment_label"] == "Negative").sum()))

st.subheader("Unified Complaint View")
filter_cols = st.columns(4)
channel_f = filter_cols[0].multiselect("Channel", sorted(processed_df["channel"].unique()))
category_f = filter_cols[1].multiselect("Predicted Category", sorted(processed_df["predicted_category"].unique()))
region_f = filter_cols[2].multiselect("Region", sorted(processed_df["region"].unique()))
urgency_f = filter_cols[3].multiselect("Urgency", sorted(processed_df["urgency"].unique()))

view = processed_df.copy()
if channel_f:
    view = view[view["channel"].isin(channel_f)]
if category_f:
    view = view[view["predicted_category"].isin(category_f)]
if region_f:
    view = view[view["region"].isin(region_f)]
if urgency_f:
    view = view[view["urgency"].isin(urgency_f)]

st.dataframe(
    view[
        [
            "complaint_id",
            "timestamp",
            "channel",
            "region",
            "branch",
            "complaint_text",
            "predicted_category",
            "sentiment_label",
            "urgency",
            "is_duplicate",
            "sla_status",
            "draft_response",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Trend and Insight Panels")
chart_col1, chart_col2 = st.columns(2)

by_channel = view.groupby("channel", as_index=False).size().rename(columns={"size": "count"})
fig_channel = px.bar(by_channel, x="channel", y="count", title="Complaints by Channel")
chart_col1.plotly_chart(fig_channel, use_container_width=True)

by_category = view.groupby("predicted_category", as_index=False).size().rename(columns={"size": "count"})
fig_category = px.pie(by_category, values="count", names="predicted_category", title="Category Mix")
chart_col2.plotly_chart(fig_category, use_container_width=True)

trend_df = view.copy()
trend_df["date"] = pd.to_datetime(trend_df["timestamp"]).dt.date
by_date = trend_df.groupby("date", as_index=False).size().rename(columns={"size": "count"})
fig_trend = px.line(by_date, x="date", y="count", markers=True, title="Complaint Volume Trend")
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("POC Scope: Real vs Simulated")
st.markdown(
    "- Real in POC: end-to-end data flow, NLP categorization model, sentiment scoring, urgency scoring, duplicate detection, draft response generation, and dashboarding.\n"
    "- Simulated in POC: complaint channels, customer metadata, and SLA timestamps are synthetic for safe demonstration.\n"
    "- External dataset support: upload a Kaggle-derived CSV if the PS dataset is available, after mapping it to complaint_text and category_label columns.\n"
    "- Optional live LLM: set OPENAI_API_KEY and select 'openai' provider in sidebar."
)

if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI provider selected but OPENAI_API_KEY is not set. Falling back to template responses.")

if 'insights' in locals() and insights:
    st.subheader("Root Cause Insights (from API)")
    st.json(insights)
