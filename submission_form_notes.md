# Submission form notes (use only for matching fields in the actual form)

**What did you build, and what business outcome does it move?**
Built a local service-request classifier and FastAPI + Streamlit demo to suggest one of Kestrel's seven canonical service teams. On a chronological holdout of 1,438 May–June 2026 requests, the raw classifier matched 96.31% of historical initial-queue labels; after a narrow Repairs policy override, accuracy was 93.53%. The intended business outcome is fewer misroutes, transfers and repeat customer contacts; this must be verified in shadow mode before claiming savings.

**What number do you expect it to score and why?**
96.31% raw model accuracy against historical first-team labels, using training data through 30 April 2026 and a later May–June holdout. The deployed model applies a narrow fault-to-Repairs policy override and scores 93.53% on that same holdout. This is an estimate, not a measured score on the unlabelled July–September test period. The model reproduces historical initial routing, which may itself be imperfect.

**What did you use?**
Python, pandas, scikit-learn (word/character TF-IDF + LinearSVC), joblib, FastAPI and Streamlit. No paid model API is used.

**What did it cost?**
₹0 in paid model-API charges for this implementation. This excludes the user's development time and any hosting, monitoring or maintenance costs. The existing vendor licence is ₹3.2 lakh/year (₹26,667/month approximately); cancellation should wait until shadow-mode outcomes and total run cost are reviewed.

**What did you discard?**
A hosted LLM/RAG/multi-agent design was not justified for this seven-class labelled routing task and would add external API costs and dependencies without proven quality gains. A broad rule-only classifier was discarded; a narrow product-fault override was retained because the policy explicitly routes faults to Repairs.
