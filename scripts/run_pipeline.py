import argparse

import pandas as pd

from src.data_generator import generate_synthetic_complaints
from src.genai import generate_draft_response
from src.pipeline import process_complaints


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
    out["draft_response"] = out.apply(
        lambda r: generate_draft_response(
            complaint_text=r["complaint_text"],
            predicted_category=r["predicted_category"],
            urgency=r["urgency"],
            provider="template",
        ),
        axis=1,
    )

    print("Processed rows:", len(out))
    print(out[["complaint_id", "channel", "predicted_category", "sentiment_label", "urgency", "is_duplicate", "sla_status"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
