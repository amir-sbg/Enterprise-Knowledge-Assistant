from __future__ import annotations

import re

from rag_system.schema import Answer, Citation, RetrievedChunk
from rag_system.text import content_terms, token_set


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ExtractiveAnswerer:
    def answer(self, question: str, retrieved: list[RetrievedChunk], max_citations: int = 3) -> Answer:
        if not retrieved:
            return Answer(
                question=question,
                text="I could not find enough evidence in the indexed knowledge base to answer that.",
                citations=[],
                retrieval=[],
            )

        q_terms = content_terms(question) or token_set(question)
        min_overlap = 2 if len(q_terms) >= 4 else 1
        citations: list[Citation] = []
        answer_parts: list[str] = []

        candidates = []
        for item in retrieved:
            sentence = _best_sentence(item.chunk.text, q_terms)
            if not sentence:
                continue
            overlap = len(content_terms(sentence) & q_terms)
            if overlap >= min_overlap:
                candidates.append((overlap, item.score, item, sentence))

        if not candidates and retrieved:
            sentence = _best_sentence(retrieved[0].chunk.text, q_terms)
            candidates.append((0, retrieved[0].score, retrieved[0], sentence))

        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)

        for _, _, item, sentence in candidates[:max_citations]:
            citations.append(
                Citation(
                    chunk_id=item.chunk.id,
                    document_id=item.chunk.document_id,
                    source=str(item.chunk.metadata.get("source", item.chunk.document_id)),
                    quote=sentence,
                    score=item.score,
                )
            )
            answer_parts.append(sentence)

        if not answer_parts:
            answer_parts = [_best_sentence(retrieved[0].chunk.text, q_terms) or retrieved[0].chunk.text]

        answer_text = " ".join(answer_parts)
        answer_text += " " + " ".join(f"[{idx}]" for idx in range(1, len(citations) + 1))
        return Answer(question=question, text=answer_text.strip(), citations=citations, retrieval=retrieved)


def _best_sentence(text: str, q_terms: set[str]) -> str:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
    if not sentences:
        return text.strip()
    return max(sentences, key=lambda sentence: len(token_set(sentence) & q_terms))
