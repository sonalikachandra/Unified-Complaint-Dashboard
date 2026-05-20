# D1: Problem + Solution Brief (Draft)

## Page 1: Problem Context (Union Bank Focus)
Union Bank receives customer complaints across fragmented channels such as branch registers, email, call center notes, web forms, and social media. This fragmentation causes delayed triage, duplicate handling, low transparency for customers, and weak escalation control. Different systems and manual logs make it hard to track complaint status in one place or identify repeated root causes by branch/product.

Current approaches fail because complaint processing is largely manual and reactive. Agents spend time classifying complaints, checking history, and drafting responses. This increases turnaround time and causes SLA misses, especially in high-risk cases such as fraud or blocked accounts. Management dashboards are often delayed and not action-oriented.

## Page 2: Proposed Solution
We propose an AI-powered Unified Customer Complaint Communication Dashboard for Union Bank. The system ingests complaints from multiple channels into one queue and performs automated complaint intelligence:
- NLP auto-categorization.
- Sentiment and urgency detection.
- Duplicate issue detection.
- AI-assisted draft responses.
- SLA breach monitoring and escalation tags.
- Trends by branch, region, and complaint type.

How it works (POC):
1. Synthetic multi-channel complaint data is generated.
2. A text classification model predicts complaint category.
3. Sentiment and urgency are scored.
4. Similar complaints are flagged as duplicates.
5. Draft responses are generated using template/LLM mode.
6. Dashboard displays all records, alerts, and trends.

What makes it better:
- Faster triage and resolution readiness.
- Better consistency of responses.
- Early visibility into repeated pain points.
- Stronger operational monitoring for managers.

POC Note:
- Real: end-to-end AI pipeline, scoring logic, dashboard.
- Simulated: complaint source systems and customer data.
