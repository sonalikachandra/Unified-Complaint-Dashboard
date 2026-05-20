# PS5 Prototype Requirements Checklist

## Mandatory for Evaluation (Minimum Acceptable PS5 POC)
- Unified complaint dashboard with complaints from at least 2-3 channels.
- NLP auto-categorization pipeline.
- Sentiment scoring pipeline.
- Gen-AI complaint response drafting.
- End-to-end flow: input -> processing -> output.
- Clear real vs simulated disclosure.

## Data Requirements
- Synthetic complaint dataset with fields:
  - complaint_id, timestamp, customer_id
  - channel, region, branch, product, language
  - complaint_text, category label (for model training)
- At least 300 rows for visible trends.

## Model/Pipeline Requirements
- Text classification model for category prediction.
- Sentiment engine (rule/model based).
- Urgency scoring logic.
- Duplicate complaint detection.
- SLA breach logic and escalation tag.
- Draft response generator (template or LLM API).

## UI Requirements
- Metrics cards: volume, high urgency, duplicates, SLA breaches, negative sentiment.
- Unified complaint table with filters.
- Trend visualizations (channel/category/time).
- Draft response preview in-line.

## Engineering Requirements
- Runnable locally on laptop.
- Dependency manifest (`requirements.txt`).
- `README.md` with setup/run instructions.
- Script to generate synthetic data.
- Basic runner/harness for non-UI validation.

## Deliverable Mapping
- D1: problem and solution brief (`docs/D1_problem_solution_brief.md`).
- D2: technical demo video script (`docs/D2_demo_script.md`) + app run.
- D3: architecture document (`docs/D3_technical_architecture.md`).
- D4: GitHub repository with complete code and README.
- D5: pitch script (`docs/D5_pitch_script.md`) and final slide links.

