import argparse
import os
import sys

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_generator import generate_synthetic_complaints
from src.genai import generate_draft_response, get_last_provider_status
from src.pipeline import process_complaints

BULK_GEMINI_DRAFT_LIMIT = 3


def _load_data(input_csv: str | None) -> pd.DataFrame:
    if input_csv:
        return pd.read_csv(input_csv)
    return generate_synthetic_complaints(n=120, seed=42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PS5 complaint pipeline")
    parser.add_argument("--input-csv", dest="input_csv", default=None, help="Optional complaint CSV with complaint_text and category_label columns")
    args = parser.parse_args()

    df = _load_data(args.input_csv)
    out, _ = process_complaints(df)
    provider_statuses = []

    def make_draft(r: pd.Series) -> str:
        provider = "gemini" if r.name < BULK_GEMINI_DRAFT_LIMIT else "template"
        draft = generate_draft_response(
            complaint_text=r["complaint_text"],
            predicted_category=r["predicted_category"],
            urgency=r["urgency"],
            provider=provider,
        )
        provider_statuses.append(get_last_provider_status())
        return draft

    out["draft_response"] = out.apply(make_draft, axis=1)
    out["draft_source"] = [status.get("used", "template") for status in provider_statuses]

    print("Processed rows:", len(out))
    print(out[["complaint_id", "channel", "predicted_category", "sentiment_label", "urgency", "is_duplicate", "sla_status"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
