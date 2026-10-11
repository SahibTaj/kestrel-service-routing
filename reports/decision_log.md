# Decision log

- **Target:** historical `team_label` (initial queue), because the client explicitly asks to match the routing bot labels. `resolution_log.final_team` is retained for error/operations analysis, not substituted as the target.
- **Validation:** chronological, not random. Train on data through 2026-04-30 and hold out May–June 2026 to better mimic the newer test period.
- **Canonical labels:** map `Installs & Demo` → `Installations`; `Filters & Consumables` → `Consumables`. These are renames in `teams.csv` and the policy states responsibilities did not change.
- **Features:** request text plus channel, product family, warranty status, and source. Do not use request IDs or resolution fields as predictive features.
- **Model:** word and character TF-IDF + LinearSVC. The selected metadata-aware model achieved 96.31% holdout accuracy in the experiment. A simpler local model avoids per-request paid API costs.
- **Reasons:** use matched policy cues where possible; otherwise explain similarity to historical examples. Reasons are not claimed as causal model explanations.
- **Rollout:** shadow mode and human audit before replacing the incumbent bot.
