from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import Dict, List

import pandas as pd


CHANNELS = ["Email", "Branch", "Chat", "Call Center", "Social Media"]
REGIONS = ["North", "South", "East", "West", "Central"]
BRANCHES = ["Mumbai Main", "Lucknow Hazratganj", "Bengaluru MG Road", "Pune Camp", "Jaipur MI Road"]
PRODUCTS = ["Savings", "Current", "Credit Card", "UPI", "Loan", "Internet Banking"]
LANGUAGES = ["English", "Hindi", "Marathi"]

CATEGORY_TEMPLATES: Dict[str, List[str]] = {
    "UPI/Payments": [
        "My UPI payment of Rs {amount} failed but amount got debited.",
        "Transaction pending for more than {hours} hours in UPI app.",
        "Beneficiary not receiving money even after successful status.",
    ],
    "Cards/ATM": [
        "ATM debited cash but did not dispense money at {branch}.",
        "Credit card payment posted twice and statement is incorrect.",
        "Card blocked without notice and merchant payment failed.",
    ],
    "Loan/Credit": [
        "Loan EMI auto-debit failed despite sufficient balance.",
        "Personal loan closure not updated and extra interest charged.",
        "Credit score impacted due to incorrect overdue on loan account.",
    ],
    "Internet Banking": [
        "Unable to login to internet banking since yesterday.",
        "Password reset link keeps failing with server error.",
        "Net banking page times out while downloading statement.",
    ],
    "Account Services": [
        "Address update request pending for {days} days.",
        "Cheque book request not processed and no tracking available.",
        "Account statement email not received despite request.",
    ],
    "Fraud/Security": [
        "Unauthorized transaction detected on my account, please block immediately.",
        "Received phishing call pretending to be bank officer.",
        "Suspicious login alert came from unknown location.",
    ],
}


@dataclass
class ComplaintRecord:
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


def _render_text(template: str, rng: random.Random) -> str:
    return template.format(
        amount=rng.randint(500, 50000),
        hours=rng.randint(2, 48),
        days=rng.randint(2, 21),
        branch=rng.choice(BRANCHES),
    )


def generate_synthetic_complaints(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    base_time = datetime.now() - timedelta(days=14)

    rows: List[ComplaintRecord] = []
    categories = list(CATEGORY_TEMPLATES.keys())

    for i in range(1, n + 1):
        category = rng.choice(categories)
        text = _render_text(rng.choice(CATEGORY_TEMPLATES[category]), rng)

        if rng.random() < 0.08:
            text = text + " This issue has been repeated multiple times and still unresolved."

        rec = ComplaintRecord(
            complaint_id=f"CMP-{i:05d}",
            timestamp=(base_time + timedelta(minutes=45 * i)).isoformat(timespec="minutes"),
            customer_id=f"CUST-{rng.randint(1000, 9999)}",
            channel=rng.choice(CHANNELS),
            region=rng.choice(REGIONS),
            branch=rng.choice(BRANCHES),
            product=rng.choice(PRODUCTS),
            language=rng.choice(LANGUAGES),
            complaint_text=text,
            category_label=category,
        )
        rows.append(rec)

    return pd.DataFrame([r.__dict__ for r in rows])


def save_sample_dataset(path: str = "data/sample_complaints.csv", n: int = 350, seed: int = 42) -> str:
    df = generate_synthetic_complaints(n=n, seed=seed)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    output = save_sample_dataset()
    print(f"Synthetic complaints saved to {output}")
