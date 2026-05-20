from __future__ import annotations

import os


def _template_draft(category: str, urgency: str, complaint_text: str) -> str:
    prefix = {
        "High": "We have prioritized your complaint for immediate resolution.",
        "Medium": "Your complaint has been assigned to the concerned team.",
        "Low": "Thank you for writing to us. We have registered your request.",
    }.get(urgency, "We have received your complaint.")

    action = {
        "UPI/Payments": "We are validating transaction logs and will reverse any incorrect debit after verification.",
        "Cards/ATM": "Our card and ATM operations team is checking switch logs and branch reconciliation.",
        "Loan/Credit": "We are reviewing your loan account statement and interest/EMI entries.",
        "Internet Banking": "Our digital banking team is investigating the service disruption and access logs.",
        "Account Services": "We are verifying your service request timeline and pending fulfillment steps.",
        "Fraud/Security": "We have initiated account security protocols and flagged suspicious activity for review.",
    }.get(category, "We are reviewing your issue and will share an update.")

    return (
        f"{prefix} {action} "
        "Reference ID has been created and next update will be shared within 24 hours. "
        f"Summary captured: {complaint_text[:120]}"
    )


def generate_draft_response(
    complaint_text: str,
    predicted_category: str,
    urgency: str,
    provider: str = "template",
) -> str:
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Draft a concise bank complaint response in 4-5 lines. "
                f"Category: {predicted_category}. Urgency: {urgency}. "
                f"Customer complaint: {complaint_text}"
            )
            result = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0.2,
            )
            return result.output_text.strip()
        except Exception:
            return _template_draft(predicted_category, urgency, complaint_text)

    return _template_draft(predicted_category, urgency, complaint_text)
