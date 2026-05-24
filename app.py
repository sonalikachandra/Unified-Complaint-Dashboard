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
from src.genai import generate_draft_response, get_last_provider_status
from src.pipeline import generate_root_cause_insights, process_complaints


st.set_page_config(page_title="Unified Complaint Dashboard", layout="wide")

BULK_GEMINI_DRAFT_LIMIT = 3

st.session_state.setdefault("provider_choice", "gemini")


def suggest_action(category: str, urgency: str, sla_status: str = "Within SLA", is_duplicate: bool = False) -> str:
    if is_duplicate:
        return "Link with existing complaint and avoid duplicate handling."
    if sla_status == "Breached":
        return "Escalate to service manager and send urgent customer update."
    if urgency == "High":
        return "Assign to priority queue and contact customer immediately."

    actions = {
        "UPI/Payments": "Verify transaction status and initiate reversal if debit is confirmed.",
        "Cards/ATM": "Check ATM/card switch logs and branch reconciliation status.",
        "Loan/Credit": "Review loan ledger, EMI posting, and credit bureau impact.",
        "Internet Banking": "Route to digital banking support for access/log review.",
        "Account Services": "Check service request ageing and pending fulfilment step.",
        "Fraud/Security": "Trigger security review and block risky activity if required.",
    }
    return actions.get(category, "Assign to the concerned operations team for resolution.")

st.markdown(
    """
    <style>
    .stDataFrame, .stDataFrame * {
        font-size: 0.82rem !important;
    }
    .stDataFrame [role="gridcell"],
    .stDataFrame [role="columnheader"] {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    .live-result-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.5rem;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }
    .live-result-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 0.65rem;
        padding: 0.55rem 0.7rem;
        background: rgba(250, 250, 250, 0.85);
        min-width: 0;
    }
    .live-result-label {
        font-size: 0.7rem;
        font-weight: 600;
        opacity: 0.72;
        margin-bottom: 0.2rem;
    }
    .live-result-value {
        font-size: 0.8rem;
        line-height: 1.2;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    .live-response-box {
        font-size: 0.8rem;
        line-height: 1.35;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Unified Customer Complaint Communication Dashboard")
st.caption("POC flow: Multi-channel complaint intake -> NLP categorization -> sentiment/urgency -> duplicate detection -> Gen-AI draft responses")

# ============================================================================
# LIVE COMPLAINT ANALYZER - For Judges to Test AI Instantly
# ============================================================================
st.markdown("---")
st.subheader("Live Complaint Analyzer")
st.markdown("**Test the AI instantly:** Enter a complaint, select a channel, and see real-time NLP analysis.")
st.caption(f"Active draft provider: {st.session_state.get('provider_choice', 'gemini')}")

analyzer_col1, analyzer_col2 = st.columns([3, 1])
with analyzer_col1:
    live_complaint = st.text_area(
        "Enter a complaint",
        placeholder="e.g., Money deducted from ATM but cash not received. No response from support for 5 days.",
        height=100,
        key="live_complaint_input"
    )
with analyzer_col2:
    live_channel = st.selectbox(
        "Channel",
        ["Email", "Chat", "Call Center", "Branch", "Social Media"],
        key="live_channel_input"
    )

analyze_btn = st.button("Analyze", key="analyze_live_btn", type="primary")

if analyze_btn and live_complaint.strip():
    # Create a minimal complaint record for analysis
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    # Use existing processed_df if available, else generate minimal training set
    if "processed_df" in st.session_state and len(st.session_state["processed_df"]) > 0:
        train_df = st.session_state["processed_df"][["complaint_text", "category_label"]].drop_duplicates()
    else:
        # Fallback: generate synthetic for training
        train_df = generate_synthetic_complaints(n=100, seed=42)[["complaint_text", "category_label"]]
    
    # Train classifier on available data
    from src.pipeline import train_category_model
    artifacts = train_category_model(train_df)
    classifier = artifacts.classifier
    
    # Predict category
    live_pred = classifier.predict([live_complaint])[0]
    live_category = live_pred
    
    # Sentiment & urgency - using VADER directly
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(live_complaint)
    compound = scores["compound"]
    if compound < -0.05:
        live_sentiment_label = "Negative"
    elif compound > 0.05:
        live_sentiment_label = "Positive"
    else:
        live_sentiment_label = "Neutral"
    
    # Urgency based on negative sentiment and keywords
    urgency_keywords = ["urgent", "asap", "immediately", "critical", "issue", "problem", "deducted", "received", "denied", "refused"]
    has_urgency_keywords = any(kw in live_complaint.lower() for kw in urgency_keywords)
    if live_sentiment_label == "Negative" and has_urgency_keywords:
        live_urgency = "High"
    elif live_sentiment_label == "Negative":
        live_urgency = "Medium"
    else:
        live_urgency = "Low"
    live_sla_risk = "High" if live_urgency == "High" else "Normal"
    
    # Duplicate check against existing data
    live_duplicate_score = 0.0
    if "processed_df" in st.session_state and len(st.session_state["processed_df"]) > 0:
        dup_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        all_texts = list(st.session_state["processed_df"]["complaint_text"].values) + [live_complaint]
        dup_vecs = dup_vectorizer.fit_transform(all_texts)
        live_dup_sims = cosine_similarity(dup_vecs[-1:], dup_vecs[:-1]).flatten()
        if len(live_dup_sims) > 0:
            live_duplicate_score = round(live_dup_sims.max() * 100, 1)
    
    # Draft response
    live_draft = generate_draft_response(
        complaint_text=live_complaint,
        predicted_category=live_category,
        urgency=live_urgency,
        provider=st.session_state.get("provider_choice", "gemini")
    )
    live_provider_status = get_last_provider_status()
    
    # Display results in a clean card layout
    st.markdown("### AI Analysis Results")
    
    st.markdown(
        f"""
        <div class="live-result-grid">
            <div class="live-result-card">
                <div class="live-result-label">Category</div>
                <div class="live-result-value">{live_category}</div>
            </div>
            <div class="live-result-card">
                <div class="live-result-label">Sentiment</div>
                <div class="live-result-value">{live_sentiment_label}</div>
            </div>
            <div class="live-result-card">
                <div class="live-result-label">Urgency</div>
                <div class="live-result-value">{live_urgency}</div>
            </div>
            <div class="live-result-card">
                <div class="live-result-label">Duplicate Match</div>
                <div class="live-result-value">{live_duplicate_score}%</div>
            </div>
            <div class="live-result-card">
                <div class="live-result-label">SLA Risk</div>
                <div class="live-result-value">{live_sla_risk}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("#### Suggested Response")
    if live_provider_status.get("used") == "template" and live_provider_status.get("provider") != "template":
        print(
            f"{live_provider_status['provider']} live draft provider fallback: "
            f"{live_provider_status.get('error')}"
        )
        st.caption("AI draft service is temporarily unavailable. A standard response draft is shown for continuity.")
    st.markdown(f"<div class='live-response-box'>{live_draft}</div>", unsafe_allow_html=True)
    st.markdown("#### Suggested Action")
    st.info(suggest_action(live_category, live_urgency))
    
    st.markdown("---")

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
        ["gemini", "openai", "template"],
        key="provider_choice",
        help="OpenAI mode needs OPENAI_API_KEY. Gemini mode needs GEMINI_API_KEY or GOOGLE_API_KEY. Falls back to template if unavailable.",
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
    provider_statuses = []

    def make_draft(r: pd.Series) -> str:
        draft_provider = provider
        if provider == "gemini" and r.name >= BULK_GEMINI_DRAFT_LIMIT:
            draft_provider = "template"

        draft = generate_draft_response(
            complaint_text=r["complaint_text"],
            predicted_category=r["predicted_category"],
            urgency=r["urgency"],
            provider=draft_provider,
        )
        provider_statuses.append(get_last_provider_status())
        return draft

    processed_df["draft_response"] = processed_df.apply(make_draft, axis=1)
    processed_df["draft_source"] = [
        status.get("used", "template") for status in provider_statuses
    ]
    processed_df["suggested_action"] = processed_df.apply(
        lambda r: suggest_action(
            r["predicted_category"],
            r["urgency"],
            r["sla_status"],
            bool(r["is_duplicate"]),
        ),
        axis=1,
    )
    fallback_count = sum(
        1
        for status in provider_statuses
        if status.get("used") == "template" and status.get("provider") != "template"
    )
    fallback_errors = [
        status.get("error")
        for status in provider_statuses
        if status.get("used") == "template"
        and status.get("provider") != "template"
        and status.get("error")
    ]
    if fallback_errors:
        print(f"{provider} draft provider fallback. First error: {fallback_errors[0]}")
    if provider == "gemini" and len(processed_df) > BULK_GEMINI_DRAFT_LIMIT:
        st.caption(
            f"Draft generation optimized for demo: AI drafts are sampled for {BULK_GEMINI_DRAFT_LIMIT} rows; "
            "standard drafts are used for the remaining rows."
        )
    elif fallback_count:
        st.caption("Some response drafts used the standard fallback because the AI service was temporarily unavailable.")

    st.session_state["processed_df"] = processed_df

processed_df = st.session_state["processed_df"]
if "draft_source" not in processed_df.columns:
    processed_df["draft_source"] = "template"
if "suggested_action" not in processed_df.columns:
    processed_df["suggested_action"] = processed_df.apply(
        lambda r: suggest_action(
            r["predicted_category"],
            r["urgency"],
            r["sla_status"],
            bool(r["is_duplicate"]),
        ),
        axis=1,
    )

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
            "draft_source",
            "suggested_action",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "predicted_category": st.column_config.TextColumn("Predicted Category", width="small"),
        "complaint_text": st.column_config.TextColumn("Complaint Text", width="large"),
        "draft_response": st.column_config.TextColumn("Draft Response", width="large"),
        "draft_source": st.column_config.TextColumn("Draft Source", width="small"),
        "suggested_action": st.column_config.TextColumn("Suggested Action", width="large"),
    },
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

analytics_col1, analytics_col2 = st.columns(2)
by_sentiment = view.groupby("sentiment_label", as_index=False).size().rename(columns={"size": "count"})
fig_sentiment = px.pie(by_sentiment, values="count", names="sentiment_label", title="Sentiment Mix")
analytics_col1.plotly_chart(fig_sentiment, use_container_width=True)

by_sla = view.groupby("sla_status", as_index=False).size().rename(columns={"size": "count"})
fig_sla = px.bar(by_sla, x="sla_status", y="count", title="SLA Status")
analytics_col2.plotly_chart(fig_sla, use_container_width=True)

st.subheader("SLA Risk Cards")
risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
risk_col1.metric("High Priority Queue", int((view["urgency"] == "High").sum()))
risk_col2.metric("Breached SLA", int((view["sla_status"] == "Breached").sum()))
risk_col3.metric("Duplicate Workload", int(view["is_duplicate"].sum()))
risk_col4.metric("Gemini Drafts", int((view["draft_source"] == "gemini").sum()))

st.subheader("Duplicate Complaints")
duplicate_view = view[view["is_duplicate"]][
    [
        "complaint_id",
        "duplicate_of",
        "duplicate_cluster",
        "channel",
        "branch",
        "predicted_category",
        "complaint_text",
        "suggested_action",
    ]
]
if duplicate_view.empty:
    st.info("No duplicate complaints detected in the current filtered view.")
else:
    st.dataframe(duplicate_view, use_container_width=True, hide_index=True)

st.subheader("Root Cause Intelligence")
if view.empty:
    st.info("No complaints available for root-cause analysis in the current filtered view.")
else:
    root_insights = generate_root_cause_insights(view)
    insight_lines = []

    for category, detail in root_insights.items():
        category_df = view[view["predicted_category"] == category]
        if category_df.empty:
            continue

        top_branch = category_df["branch"].value_counts().head(1)
        if not top_branch.empty:
            branch_name = top_branch.index[0]
            branch_count = int(top_branch.iloc[0])
            branch_pct = round((branch_count / len(category_df)) * 100)
            insight_lines.append(
                f"{branch_pct}% of {category} complaints are concentrated at {branch_name}."
            )

        top_phrases = detail.get("top_phrases", [])
        if top_phrases:
            phrase, count = top_phrases[0]
            insight_lines.append(
                f"Frequent issue phrase for {category}: '{phrase}' appeared {count} times."
            )

    hour_df = view.copy()
    hour_df["hour"] = pd.to_datetime(hour_df["timestamp"]).dt.hour
    busiest_hour = hour_df.groupby("hour").size().sort_values(ascending=False).head(1)
    if not busiest_hour.empty:
        hour = int(busiest_hour.index[0])
        insight_lines.append(
            f"Complaint spike detected around {hour:02d}:00-{(hour + 1) % 24:02d}:00."
        )

    for line in insight_lines[:6]:
        st.markdown(f"- {line}")

st.subheader("POC Scope: Real vs Simulated")
st.markdown(
    "- Real in POC: end-to-end data flow, NLP categorization model, sentiment scoring, urgency scoring, duplicate detection, draft response generation, and dashboarding.\n"
    "- Simulated in POC: complaint channels, customer metadata, and SLA timestamps are synthetic for safe demonstration.\n"
    "- External dataset support: upload a Kaggle-derived CSV if the PS dataset is available, after mapping it to complaint_text and category_label columns.\n"
    "- Optional live LLM: set OPENAI_API_KEY or GEMINI_API_KEY and select the matching provider in sidebar."
)

if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
    st.caption("OpenAI draft mode is unavailable because the API key is not configured.")
elif provider == "gemini" and not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.caption("Gemini draft mode is unavailable because the API key is not configured.")

if 'insights' in locals() and insights:
    st.subheader("Root Cause Insights (from API)")
    st.json(insights)
