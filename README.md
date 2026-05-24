# PS5 - Unified Customer Complaint Communication Dashboard (POC)

This repository contains a working Proof of Concept (POC) for PS5 (iDEA 2.0):
- Unified complaint intake from multiple simulated channels.
- NLP auto-categorization.
- Sentiment and urgency scoring.
- Duplicate complaint detection.
- Gen-AI draft response generation.
- Dashboard for operations and management insights.

## 1. Problem Being Solved
Union Bank complaint workflows are fragmented across channels. This causes slower resolution, duplicate handling, and weaker SLA monitoring. The POC demonstrates a unified AI-assisted dashboard to improve triage and response quality.

## 2. What Is Built vs Simulated
Built now:
- End-to-end processing pipeline.
- Trained complaint category classifier.
- Sentiment and urgency logic.
- Duplicate detection.
- Draft response generation (template + optional OpenAI or Gemini mode).
- Streamlit dashboard with filters and trends.
- CSV upload support so an external dataset can be used when available.

Simulated:
- Complaint source channels and customer data are synthetic.
- SLA timelines are demo logic.

## 3. Project Structure
- `app.py`: Streamlit dashboard.
- `src/data_generator.py`: synthetic multi-channel complaint dataset generator.
- `src/pipeline.py`: NLP and scoring pipeline.
- `src/genai.py`: draft response generation.
- `scripts/run_pipeline.py`: command-line runner for quick validation.
- `docs/`: D1, D2, D3, D5 draft materials and requirement extraction.

## 4. Setup
```powershell
cd c:\Users\Hp\OneDrive\Desktop\IDEA
C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe -m pip install -r requirements.txt
```

## 5. Run Locally
Run pipeline check:
```powershell
cd c:\Users\Hp\OneDrive\Desktop\IDEA
C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe -m scripts.run_pipeline
```

Run dashboard:
```powershell
cd c:\Users\Hp\OneDrive\Desktop\IDEA
C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run app.py
```

## 6. Optional LLM Draft Responses
If you want LLM-generated draft replies instead of template mode:
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run app.py
```
Or use Gemini:
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
C:/Users/Hp/AppData/Local/Programs/Python/Python311/python.exe -m streamlit run app.py
```
If your Google key is stored under `GOOGLE_API_KEY`, that works too.
Then choose `openai` or `gemini` in the sidebar provider dropdown.

## 7. Dependencies
See `requirements.txt`.

## 8. Sample Dataset
Use synthetic generator in `src/data_generator.py`.
Example output file path (optional): `data/sample_complaints.csv`.
If a Kaggle complaint dataset is available for the PS, preprocess or map it to at least `complaint_text` and `category_label`, then upload it in the dashboard.

## 9. Deployment
The app currently runs locally on Streamlit. After hosting it on Streamlit Community Cloud, Render, or another platform, paste the public deployment URL into your final submission materials and end slide.

## FastAPI Backend (Optional)
A minimal FastAPI backend is included at `src/api.py` to demonstrate live processing endpoints and root-cause insights.

Run the API server (recommended port `8000`):
```powershell
cd c:\Users\Hp\OneDrive\Desktop\IDEA
# install FastAPI and uvicorn if not installed
python -m pip install fastapi uvicorn
# start server
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /upload-csv` — multipart file upload (CSV) to process and store results.
- `POST /ingest` — JSON complaint object to append and reprocess.
- `GET /processed` — returns processed complaint records.
- `GET /insights` — returns root-cause style insights (top branches, phrases by category).

Note: on this machine, dependency installation previously failed due to insufficient disk space when running `pip install -r requirements.txt`. If you hit that error, free some disk space and re-run the install commands above.

## 10. Known Limitations
- Dataset is synthetic and not multilingual-rich yet.
- No live banking API integrations.
- Duplicate detection is lexical similarity based (not semantic embeddings).
- LLM response quality depends on provider/API availability.

## 11. Submission-Ready Docs
- D1 draft: `docs/D1_problem_solution_brief.md`
- D2 script: `docs/D2_demo_script.md`
- D3 draft: `docs/D3_technical_architecture.md`
- D5 script: `docs/D5_pitch_script.md`
- PS extraction and requirements: `docs/ps5_extraction.md`, `docs/prototype_requirements.md`
