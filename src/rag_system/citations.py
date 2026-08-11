from __future__ import annotations

from dataclasses import dataclass

from rag_system.schema import Answer
from rag_system.text import token_set


@dataclass(slots=True)
class CitationReport:
    citation_accuracy: float
    faithfulness: float
    unsupported_claims: list[str]
    missing_citations: list[str]


class CitationVerifier:
    def verify(self, answer: Answer) -> CitationReport:
        if not answer.text:
            return CitationReport(0.0, 0.0, [], [])
        if not answer.citations:
            return CitationReport(0.0, 0.0, [answer.text], ["answer has no citations"])

        citation_text = " ".join(citation.quote for citation in answer.citations)
        evidence_terms = token_set(citation_text)

        claims = _split_claims(answer.text)
        supported = 0
        unsupported: list[str] = []
        for claim in claims:
            claim_terms = token_set(_strip_citation_markers(claim))
            if not claim_terms:
                continue
            overlap = len(claim_terms & evidence_terms) / max(len(claim_terms), 1)
            if overlap >= 0.45:
                supported += 1
            else:
                unsupported.append(claim)

        citation_accuracy = _citation_marker_accuracy(answer.text, len(answer.citations))
        faithfulness = supported / max(len(claims), 1)
        return CitationReport(citation_accuracy, faithfulness, unsupported, [])


def _split_claims(text: str) -> list[str]:
    pieces = [piece.strip() for piece in text.split(".") if piece.strip()]
    return [piece for piece in pieces if not piece.startswith("[")]


def _strip_citation_markers(text: str) -> str:
    for marker in ["[1]", "[2]", "[3]", "[4]", "[5]"]:
        text = text.replace(marker, "")
    return text.strip()


def _citation_marker_accuracy(text: str, citation_count: int) -> float:
    if citation_count == 0:
        return 0.0
    expected = {f"[{idx}]" for idx in range(1, citation_count + 1)}
    present = {marker for marker in expected if marker in text}
    return len(present) / len(expected)

