# Decision memo: Kestrel service-request routing

**To:** Ritu Deshpande, Head of D2C Operations  
**Decision:** Proceed to a controlled shadow pilot; do not switch the vendor bot off yet.

## The number

A chronological holdout (training through 30 April 2026 and validating on May–June 2026) achieved **96.31% raw model accuracy** against historical first-team labels. We added one narrow policy guardrail: if the model predicts Consumables but the message clearly reports a product fault, the route is changed to Repairs. Validation accuracy after that adjustment was **93.53%**, still above the 90% target. This is encouraging but is not a guarantee of July–September test accuracy, nor proof that the original routing decisions were always correct. The final unlabelled requests have no outcomes available for independent scoring.

## The rupees

The current licence costs **₹3.2 lakh per year**, or approximately **₹26,667 per month**. The policy estimates ₹305 per team transfer and another ₹260 for an extra customer contact caused by a misroute. Those costs make routing quality operationally important, but we should not claim savings from classifier accuracy alone: a shadow pilot must measure actual transfers and repeat contacts against the existing process. The proposed local model uses no paid model API, so its direct model-API spend is ₹0; hosting, maintenance and staff review still have costs.

## What we built

A compact local classifier using word- and character-level text features plus channel, product, warranty and source metadata, with a narrow policy override for product faults misclassified as Consumables. It exposes one JSON endpoint and a simple screen, returns a team suggestion with human-readable routing cues, and produces the required `predictions.csv`. It runs without a paid API key.

## What changed and what we discarded

We compared a text-only TF-IDF baseline with one that also uses request metadata; the metadata-aware linear SVM performed best in the chronological validation (96.31% raw accuracy). We then measured the Repairs policy guardrail separately; it reduced agreement with historical labels to 93.53%, while preserving the explicit policy distinction for product faults. We did not add a hosted LLM, RAG, or multi-agent workflow: the task is a seven-class routing problem with labelled examples, and an external API would add variable cost and failure modes without demonstrated benefit. We normalized the two renamed teams to the seven canonical names required for submission.

## Next week

1. Run the service in shadow mode beside the current bot; do not auto-transfer customers yet.
2. Review errors weekly, especially Billing vs. Repairs/Installations and Consumables vs. Repairs.
3. Track team-level precision/recall, transfer rate, repeat contacts, latency, uptime and monthly operating cost.
4. Ask agents to correct uncertain predictions and use those adjudicated labels for the next evaluation.
5. Retire the licence only after the agreed service-level target is met on recent, independently audited requests and Finance signs off on total run cost.

**Caveat:** The model is trained to reproduce historical initial queue assignments. The resolution log shows that 2,696 of 10,822 historical requests had one or more transfers. Matching the old label is therefore a useful benchmark, not a substitute for measuring better routing outcomes.
