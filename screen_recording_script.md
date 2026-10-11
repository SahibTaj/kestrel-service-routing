# 3-minute screen recording outline

**0:00–0:25 — Business problem**  
“Kestrel wants to replace a ₹3.2 lakh/year routing bot. The dataset has 10,822 labelled requests and 2,178 newer unlabelled requests. The labels represent the initial queue, while the resolution log shows transfers, so I evaluated both label agreement and policy risks.”

**0:25–1:05 — Evidence**  
“Using a chronological holdout—training through April 30 and testing on May–June—the raw text-plus-metadata classifier reached 96.31% accuracy. I then added one narrow policy guardrail for product faults that the model otherwise routes to Consumables. The combined system reached 93.53%, above the 90% target. These are validation results, not measured accuracy on the unlabelled July–September requests.”

**1:05–1:55 — Demo**  
“Here is the Streamlit screen. I enter a water purifier leak. The model's historical-label prediction can confuse purifier vocabulary with consumables, so the policy guardrail routes a clear fault to Repairs and explains why. The FastAPI endpoint returns the team and reasons as JSON.”

**1:55–2:30 — What changed / discarded**  
“I normalized the two renamed teams to the seven canonical names. I discarded a broad rule-only approach and a hosted LLM design: the local classifier is cheaper and measurable, while the narrow policy override addresses a documented operational risk.”

**2:30–3:00 — Decision**  
“I recommend shadow mode, not immediate shutdown. Compare routes with agent corrections, track transfers and repeat contacts, audit team-level errors, and confirm total operating cost before cancelling the vendor licence.”
