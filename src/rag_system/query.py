from __future__ import annotations

import re


ACRONYM_EXPANSIONS = {
    "pii": "personally identifiable information",
    "sso": "single sign on",
    "sla": "service level agreement",
    "rag": "retrieval augmented generation",
}


class QueryRewriter:
    def rewrite(self, query: str) -> str:
        text = query.strip()
        text = re.sub(r"\s+", " ", text)
        expansions = []
        for acronym, expanded in ACRONYM_EXPANSIONS.items():
            if re.search(rf"\b{re.escape(acronym)}\b", text, re.IGNORECASE):
                expansions.append(expanded)
        if expansions:
            text = f"{text} {' '.join(expansions)}"
        return text

