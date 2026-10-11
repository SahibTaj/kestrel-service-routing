"""Reproduce chronological validation and train the final local model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from routing_policy import apply_policy_override


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REP = ROOT / "reports"

ALIASES = {
    "Installs & Demo": "Installations",
    "Filters & Consumables": "Consumables",
}

CANONICAL = [
    "Installations",
    "Repairs",
    "Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]


def input_text(df):
    return (
        df.request_text.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        + " [CHANNEL] "
        + df.channel.fillna("").astype(str)
        + " [PRODUCT] "
        + df.product_family.fillna("").astype(str)
        + " [WARRANTY] "
        + df.warranty_status.fillna("").astype(str)
        + " [SOURCE] "
        + df.source.fillna("").astype(str)
    )


def build_model():
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=180000,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=180000,
                    strip_accents="unicode",
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", LinearSVC(C=1.0, random_state=42)),
        ]
    )


def main():
    df = pd.read_csv(DATA / "train.csv")
    df["team"] = df.team_label.replace(ALIASES)

    dt = pd.to_datetime(df.created_at_ist, errors="coerce")
    mask = dt <= pd.Timestamp("2026-04-30 23:59:59")

    model = build_model()
    model.fit(input_text(df.loc[mask]), df.loc[mask, "team"])

    raw_pred = model.predict(input_text(df.loc[~mask]))
    y = df.loc[~mask, "team"]
    pred = [
        apply_policy_override(text, label)[0]
        for text, label in zip(df.loc[~mask, "request_text"], raw_pred)
    ]

    raw_acc = accuracy_score(y, raw_pred)
    final_acc = accuracy_score(y, pred)

    print("Historical-label model accuracy:", round(raw_acc, 4))
    print("After policy guardrail accuracy:", round(final_acc, 4))
    print(classification_report(y, pred, labels=CANONICAL, zero_division=0))

    REP.mkdir(exist_ok=True)
    validation_report = (
        "Chronological validation: train through 2026-04-30; "
        "validate 2026-05-01 through 2026-06-30\n"
        f"Train rows: {int(mask.sum())}\n"
        f"Validation rows: {int((~mask).sum())}\n"
        f"Raw model accuracy: {raw_acc:.4%}\n"
        f"After policy guardrail accuracy: {final_acc:.4%}\n\n"
        + classification_report(y, pred, labels=CANONICAL, zero_division=0, digits=4)
    )
    (REP / "validation_report.txt").write_text(
        validation_report,
        encoding="utf-8",
    )

    val = df.loc[~mask].copy()
    val_results = val[
        ["request_id", "created_at_ist", "request_text", "team"]
    ].copy()
    val_results["model_team"] = raw_pred
    val_results["predicted_team"] = pred
    val_results["correct"] = val_results.team == val_results.predicted_team
    val_results.loc[~val_results.correct].to_csv(
        REP / "validation_errors.csv",
        index=False,
    )

    confusion = pd.DataFrame(
        confusion_matrix(y, pred, labels=CANONICAL),
        index=CANONICAL,
        columns=CANONICAL,
    )
    confusion.to_csv(REP / "confusion_matrix.csv", index_label="actual_team")

    model = build_model()
    model.fit(input_text(df), df.team)
    ART.mkdir(exist_ok=True)
    joblib.dump(model, ART / "router.joblib", compress=3)

    test = pd.read_csv(DATA / "test_unlabelled.csv")
    raw_test = model.predict(input_text(test))
    final_test = [
        apply_policy_override(text, label)[0]
        for text, label in zip(test.request_text, raw_test)
    ]
    pd.DataFrame(
        {"request_id": test.request_id, "team": final_test}
    ).to_csv(ROOT / "predictions.csv", index=False)

    team_volume = (
        pd.Series(final_test)
        .value_counts()
        .reindex(CANONICAL, fill_value=0)
        .rename_axis("team")
        .reset_index(name="predicted_requests")
    )
    team_volume.to_csv(REP / "predicted_team_volume.csv", index=False)

    metrics = json.loads(
        (REP / "validation_metrics.json").read_text(encoding="utf-8")
    )
    metrics["raw_model_validation_accuracy"] = float(raw_acc)
    metrics["validation_accuracy"] = float(final_acc)
    metrics["validation_error_count"] = int(
        sum(a != b for a, b in zip(y, pred))
    )

    _, _, f1, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=CANONICAL,
        average="macro",
        zero_division=0,
    )
    metrics["validation_macro_f1"] = float(f1)

    _, _, f1w, _ = precision_recall_fscore_support(
        y,
        pred,
        labels=CANONICAL,
        average="weighted",
        zero_division=0,
    )
    metrics["validation_weighted_f1"] = float(f1w)
    metrics["policy_guardrail"] = (
        "When the model predicts Consumables but explicit product-fault language "
        "appears, route to Repairs. This rule was evaluated on the same "
        "chronological holdout."
    )
    metrics["policy_override_validation_count"] = int(
        sum(a != b for a, b in zip(raw_pred, pred))
    )
    (REP / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Saved", ART / "router.joblib", "and updated predictions.csv")


if __name__ == "__main__":
    main()
