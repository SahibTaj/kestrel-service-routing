# Kestrel Home — Service Request Router

A local, no-paid-API baseline that classifies each service request into one of Kestrel's seven canonical teams. The model combines word and character TF-IDF features with channel, product, warranty and source metadata, then uses a linear SVM. The service is intentionally small and starts without any external model API key.

## Validation result

Chronological holdout: train on requests dated through **30 April 2026**; validate on **1 May–30 June 2026**. See `reports/validation_report.txt`, `reports/validation_metrics.json`, `reports/confusion_matrix.csv`, and `reports/validation_errors.csv`.

This measures agreement with historical **initial queue labels**, not independent ground truth that the initial route was correct. The resolution log shows transfers; do not treat the vendor's labels as perfect operational outcomes. The unlabelled test period is newer (July–September 2026), so its true accuracy is unknown until outcomes become available.

## Team-name decision

The submission requires one of the seven canonical `teams.csv` team names. Therefore, `Installs & Demo` is normalized to `Installations`, and `Filters & Consumables` to `Consumables`. The teams' responsibilities did not change when they were renamed. This is documented so downstream comparisons are consistent.

## Files

- `predictions.csv`: required `request_id,team` output for every row in the supplied unlabelled test file.
- `artifacts/router.joblib`: fitted local model.
- `router.py`: prediction and explanation function.
- `app.py`: FastAPI service (`/health`, `/predict`).
- `ui.py`: Streamlit screen calling the API.
- `train_eval.py`: reproduce the chronological validation and train a final model.
- `reports/`: metrics, confusion matrix, error rows, data QA and predicted queue volume.
- `memo_to_ritu.md`: one-page non-technical decision memo.

## Run from a clean machine

Use Python 3.11 or 3.12. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

In a second terminal, activate the same environment and run:

```bash
streamlit run ui.py
```

Open the local Streamlit URL. API docs are at `http://127.0.0.1:8000/docs`.

To retrain from the included data:

```bash
python train_eval.py
```

## API example

```bash
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"request_id":"DEMO-1","channel":"chat","product_family":"Water Purifier","warranty_status":"in_warranty","request_text":"Water purifier is leaking from the bottom","source":"crm"}'
```

The response includes the suggested team, whether a policy override was applied, and short cue-based reasons. These are human-readable supporting cues, not causal explanations or calibrated probabilities. Empty text and a missing model are handled explicitly. For a production rollout, add authentication, request logging with data minimisation, monitoring, confidence calibration, and a human-review threshold based on measured error costs.

## Known limitations and safe rollout

1. The classifier learns the historical first-team label; the explicit Repairs override is applied only when the model predicts Consumables and the text clearly reports a product fault. It is not trained to predict the final resolving team.
2. Keyword-based explanation cues are deliberately lightweight and may not capture every nuance. Treat the output as a suggestion until shadow-mode performance is established.
3. Billing is policy-sensitive: a customer merely saying they paid is not sufficient; the underlying problem must concern the payment itself.
4. Run a shadow pilot, audit errors by team and channel, collect corrected labels, and only then consider retiring the current bot.
5. Do not publish the source data or repository publicly; the policy classifies the data as internal.
