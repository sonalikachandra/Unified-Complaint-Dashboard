from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

from src.pipeline import process_complaints, generate_root_cause_insights

app = FastAPI(title="PS5 Complaint API")

# in-memory store for demo
_raw_df: pd.DataFrame | None = None
_processed_df: pd.DataFrame | None = None


class ComplaintIn(BaseModel):
    complaint_id: str
    timestamp: str
    customer_id: str
    channel: str
    region: str
    branch: str
    product: str
    language: str
    complaint_text: str
    category_label: str


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global _raw_df, _processed_df
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    contents = await file.read()
    _raw_df = pd.read_csv(pd.io.common.BytesIO(contents))
    _processed_df, _ = process_complaints(_raw_df)
    return {"status": "processed", "rows": len(_processed_df)}


@app.post("/ingest")
async def ingest(record: ComplaintIn):
    global _raw_df, _processed_df
    row = pd.DataFrame([record.dict()])
    if _raw_df is None:
        _raw_df = row
    else:
        _raw_df = pd.concat([_raw_df, row], ignore_index=True)
    _processed_df, _ = process_complaints(_raw_df)
    return {"status": "ok", "total_rows": len(_processed_df)}


@app.get("/processed")
async def get_processed() -> List[dict]:
    if _processed_df is None:
        return []
    return _processed_df.to_dict(orient="records")


@app.get("/insights")
async def get_insights():
    if _processed_df is None:
        return {"error": "no data"}
    insights = generate_root_cause_insights(_processed_df)
    return insights
