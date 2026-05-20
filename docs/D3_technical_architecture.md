# D3: Technical Architecture (Draft)

## Components
1. Data Ingestion Layer
- Synthetic complaint generator creates records representing Email, Chat, Call Center, Branch, and Social channels.

2. NLP Intelligence Layer
- Category model: TF-IDF + Logistic Regression.
- Sentiment scoring: VADER sentiment analyzer.
- Urgency scoring: hybrid rule logic using sentiment + risk keywords.
- Duplicate detection: TF-IDF cosine similarity thresholding.

3. Gen-AI Drafting Layer
- Template draft generation (default).
- Optional OpenAI API mode for richer drafted responses.

4. Monitoring and SLA Layer
- Age calculation from complaint timestamp.
- SLA status classification (Within SLA / Breached for high urgency).

5. Presentation Layer
- Streamlit dashboard.
- Unified complaint table, filter panel, KPI cards, and trends.

## Data Flow
Input complaints -> preprocessing -> category prediction -> sentiment/urgency -> duplicate detection -> draft response generation -> dashboard rendering.

## Technical Choices
- Streamlit for rapid POC UI and deployment speed.
- Scikit-learn for transparent, lightweight classification baseline.
- Rule + ML hybrid to reduce complexity and improve explainability.
- Synthetic data for compliance-safe demonstrations.

## Security and Risk Notes
- No real customer PII used in POC.
- For production: encryption, access controls, audit logs, and human review for high-risk drafts.

## Optional Production Extensions
- Replace synthetic ingestion with real APIs (CRM, email, social connectors).
- Add multilingual pipeline and RAG-based policy grounding.
- Add root-cause and complaint surge prediction modules.
