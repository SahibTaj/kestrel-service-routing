import re
from pathlib import Path

import joblib
import pandas as pd

from routing_policy import apply_policy_override


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "router.joblib"
_model = None

TEAM_GUIDANCE = {
    "Billing": [
        r"\b(invoice|gst|double charge|refund|emi conversion|coupon|payment failed|charged twice)\b"
    ],
    "Installations": [
        r"\b(install|installation|demo|wall.?mount|mounting|installation slot)\b"
    ],
    "Consumables": [
        r"\b(filter|membrane|candle|jar|brush|blade|spare|consumable|amc kit)\b"
    ],
    "Repairs": [
        r"\b(broken|breakdown|leak|leaking|not working|stopped working|error code|fault|noise|burnt smell|gone blank)\b"
    ],
    "Returns & Replacement": [
        r"\b(return|replacement|wrong item|damaged|missing item|incomplete delivery|return pickup|exchange)\b"
    ],
    "Warranty Claims": [
        r"\b(warranty|shield plan|claim status|coverage|warranty certificate|extend warranty)\b"
    ],
    "Product Advice": [
        r"\b(how to|how do i|difference between|price|features|can .* run|usage|instructions)\b"
    ],
}


def load_model():
    global _model

    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

    return _model


def make_input(record):
    return (
        f"{str(record.get('request_text', '') or '').strip()} "
        f"[CHANNEL] {record.get('channel', '') or ''} "
        f"[PRODUCT] {record.get('product_family', '') or ''} "
        f"[WARRANTY] {record.get('warranty_status', '') or ''} "
        f"[SOURCE] {record.get('source', 'crm') or 'crm'}"
    )


def predict_record(record):
    model = load_model()

    if model is None:
        return {
            "error": "Model artifact is missing. Run `python train_eval.py` first.",
            "code": "MODEL_NOT_FOUND",
        }

    text = str(record.get('request_text', '') or '')
    if not text.strip():
        return {
            "error": "request_text must not be empty",
            "code": "INVALID_INPUT",
        }

    row = make_input(record)
    model_team = str(model.predict([row])[0])
    team, override_reason = apply_policy_override(text, model_team)
    lower = text.lower()
    reasons = []
    flags = []

    if override_reason:
        reasons.append(override_reason)

    payment_issue = bool(
        re.search(
            r"\b(invoice|gst|double charge|refund|emi conversion|coupon|payment failed|charged twice|incorrect charge)\b",
            lower,
        )
    )
    paid_only = bool(
        re.search(r"\b(already paid|paid in full|paid by|payment done)\b", lower)
    ) and not payment_issue
    fault = bool(
        re.search(
            r"\b(broken|breakdown|leak|leaking|not working|stopped working|error code|fault|noise|burnt smell|gone blank|not turning on|won't turn on)\b",
            lower,
        )
    )
    spare = bool(
        re.search(
            r"\b(filter|membrane|candle|jar|brush|blade|spare part|consumable|amc kit)\b",
            lower,
        )
    )
    install = bool(
        re.search(
            r"\b(install|installation|demo|wall.?mount|mounting|installation slot)\b",
            lower,
        )
    )
    warranty_question = bool(
        re.search(
            r"\b(warranty certificate|warranty registration|extend warranty|shield plan|claim status|covered under warranty|warranty coverage)\b",
            lower,
        )
    )

    if team == "Billing" and paid_only:
        flags.append(
            "The message only says the customer paid; policy says this alone is not a Billing issue."
        )
    if team == "Billing" and not payment_issue:
        flags.append(
            "No clear payment-problem cue was detected; verify against the Billing policy before routing."
        )
    if team != "Billing" and payment_issue and not (fault or install):
        flags.append(
            "A payment-specific cue appears in the message; verify whether Billing should own the request."
        )
    if team == "Consumables" and fault and not re.search(
        r"\b(replacement filter|filter replacement|need a filter|buy .*filter|filter price)\b",
        lower,
    ):
        flags.append(
            "A product fault is mentioned. Policy routes faults requiring repair to Repairs, even when a purifier or consumable is mentioned."
        )
    if team == "Installations" and fault:
        flags.append(
            "The message contains both installation and fault language; review whether a technician repair is the primary intent."
        )
    if team == "Repairs" and warranty_question and not fault:
        flags.append(
            "The message appears to ask about warranty or claim status rather than report a product fault; verify Warranty Claims routing."
        )

    for pattern in TEAM_GUIDANCE.get(team, []):
        match = re.search(pattern, lower)
        if match:
            reasons.append(
                f"Request text contains the cue '{match.group(0)}', which is consistent with {team} responsibilities."
            )
            break
    if not reasons:
        reasons.append(
            f"The text and request metadata are most similar to historical requests assigned to {team}."
        )
    if team == "Consumables":
        reasons.append(
            "Policy distinction: spare parts and consumables are separate from product faults requiring a technician."
        )
    if team == "Billing":
        reasons.append(
            "Policy distinction: Billing is for a problem with the payment itself, not merely a statement that the customer paid."
        )
    if flags:
        reasons.extend(flags)

    return {
        "request_id": record.get("request_id"),
        "team": team,
        "model_team": model_team,
        "policy_override": bool(override_reason),
        "reasons": reasons,
        "review_required": bool(flags),
        "policy_flags": flags,
        "model": "TF-IDF + LinearSVC",
        "note": (
            "Reasons are cue-based explanations, not causal proof. "
            "Review ambiguous or conflicting cases."
        ),
    }
