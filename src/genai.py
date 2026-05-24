from __future__ import annotations

import os
from pathlib import Path

LAST_PROVIDER_STATUS: dict[str, str | None] = {
    "provider": None,
    "used": None,
    "error": None,
}


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except Exception:
        return


_load_env_file()


def _get_api_key(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


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


def _set_provider_status(provider: str, used: str, error: str | None = None) -> None:
    LAST_PROVIDER_STATUS.update({"provider": provider, "used": used, "error": error})


def get_last_provider_status() -> dict[str, str | None]:
    return LAST_PROVIDER_STATUS.copy()


def _gemini_draft(complaint_text: str, predicted_category: str, urgency: str) -> str:
    api_key = _get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    from google import genai

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        "Draft a concise bank complaint response in 4-5 lines. "
        f"Category: {predicted_category}. Urgency: {urgency}. "
        f"Customer complaint: {complaint_text}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned no text")
    return text.strip()


def generate_draft_response(
    complaint_text: str,
    predicted_category: str,
    urgency: str,
    provider: str = "gemini",
) -> str:
    provider = (provider or "template").strip().lower()

    if provider == "gemini" and _get_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            draft = _gemini_draft(complaint_text, predicted_category, urgency)
            _set_provider_status("gemini", "gemini")
            return draft
        except Exception as exc:
            _set_provider_status("gemini", "template", f"{type(exc).__name__}: {exc}")
            return _template_draft(predicted_category, urgency, complaint_text)

    if provider == "openai" and _get_api_key("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Draft a concise bank complaint response in 4-5 lines. "
                f"Category: {predicted_category}. Urgency: {urgency}. "
                f"Customer complaint: {complaint_text}"
            )
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            draft = result.choices[0].message.content.strip()
            _set_provider_status("openai", "openai")
            return draft
        except Exception as exc:
            _set_provider_status("openai", "template", f"{type(exc).__name__}: {exc}")
            return _template_draft(predicted_category, urgency, complaint_text)

    if provider == "gemini":
        _set_provider_status("gemini", "template", "GEMINI_API_KEY / GOOGLE_API_KEY not set")
    elif provider == "openai":
        _set_provider_status("openai", "template", "OPENAI_API_KEY not set")
    else:
        _set_provider_status(provider, "template")

    return _template_draft(predicted_category, urgency, complaint_text)
