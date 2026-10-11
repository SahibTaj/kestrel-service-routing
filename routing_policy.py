"""Small, explicit policy guardrails applied after the historical-label model."""

import re

FAULT_RE = re.compile(
    r"\b(leak(?:ing)?|not working|stopped working|breakdown|error code|fault|"
    r"burnt smell|gone blank|won't turn on|wont turn on|not turning on|"
    r"making noise|loud noise|overheating|tripping the mcb|sparks?)\b",
    re.IGNORECASE,
)


def apply_policy_override(text: str, predicted_team: str):
    """Return (final_team, override_reason_or_None).

    Policy: product faults requiring a technician belong in Repairs, even if the
    text also mentions purifier filters/consumables. We limit the override to
    the known failure mode where the classifier predicts Consumables; other
    team conflicts are surfaced for review rather than silently rewritten.
    """
    if predicted_team == "Consumables" and FAULT_RE.search(text or ""):
        reason = (
            "Policy override: the message reports a product fault. Route to Repairs; "
            "a fault takes precedence over purifier/consumable vocabulary."
        )
        return "Repairs", reason

    return predicted_team, None
