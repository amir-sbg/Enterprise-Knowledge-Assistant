---
department: ai-platform
sensitivity: internal
---

# Model Governance Notes

Production AI systems must be evaluated before launch and after major retrieval, prompt, model, or data changes. The evaluation report should include task quality, retrieval quality, latency, cost, safety checks, and known failure modes.

RAG answers must cite retrieved sources when answering policy or compliance questions. If the system cannot find enough evidence, it should say that the answer is not supported by the indexed knowledge base.

For sensitive domains, hallucination checks should compare answer claims against cited chunks. Claims without supporting evidence should be marked as ungrounded and reviewed before rollout.

Release approval requires a rollback plan, monitoring dashboard, owner, and alert thresholds for latency, error rate, empty retrieval, and citation verification failures.
