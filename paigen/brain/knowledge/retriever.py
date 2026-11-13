"""
PaiGen AI: Cited Knowledge Store & Local Document Retriever
Architect: Arjun Singh

Maintains a local in-memory inverted index over PaiGen's verified knowledge base
(History, Mathematics, Philosophy, Biology, Human Condition).
Retrieves relevant documents with explicit, transparent citation tags [Citation: ...].
100% offline, zero cloud calls, fast deterministic execution.
"""

import math
import re
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from paigen.brain.knowledge.world_history import WorldHistoryRegistry
from paigen.brain.knowledge.mathematics import MathematicsRegistry
from paigen.brain.knowledge.philosophy_and_creativity import PhilosophyAndCreativityRegistry
from paigen.brain.knowledge.biology import BiologyRegistry
from paigen.brain.knowledge.human_condition import HumanConditionRegistry


class CitedDocument(BaseModel):
    doc_id: str
    citation_tag: str
    domain: str  # "History", "Mathematics", "Philosophy", "Biology", "HumanCondition"
    title: str
    content: str
    keywords: Set[str] = Field(default_factory=set)


class RetrievalResult(BaseModel):
    document: CitedDocument
    score: float
    snippet: str


class LocalKnowledgeRetriever:
    """Indexes local verified knowledge and retrieves cited evidence."""

    def __init__(self):
        self.documents: Dict[str, CitedDocument] = {}
        self.index: Dict[str, Set[str]] = {}
        self._build_index()

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
        return set(words)

    def _build_index(self):
        # 1. World History
        for ev in WorldHistoryRegistry.get_all():
            doc_id = f"hist_{ev.event_id}"
            content = f"{ev.title}. Context: {ev.core_context}. Mistakes: {' '.join(ev.critical_mistakes)}. Lessons: {' '.join(ev.enduring_lessons)}"
            tokens = self._tokenize(content)
            self.documents[doc_id] = CitedDocument(
                doc_id=doc_id,
                citation_tag=f"[Citation: WorldHistory/{ev.event_id}]",
                domain="History",
                title=ev.title,
                content=content,
                keywords=tokens
            )

        # 2. Mathematics
        for m in MathematicsRegistry.get_all():
            doc_id = f"math_{m.domain_id}"
            content = f"{m.name} ({m.category}). Titans: {', '.join(m.titans)}. Equations: {' '.join(m.core_theorems_and_equations)}. Intuition: {m.intuitive_breakthrough}"
            tokens = self._tokenize(content)
            self.documents[doc_id] = CitedDocument(
                doc_id=doc_id,
                citation_tag=f"[Citation: Mathematics/{m.domain_id}]",
                domain="Mathematics",
                title=m.name,
                content=content,
                keywords=tokens
            )

        # 3. Biology
        for b in BiologyRegistry.get_all():
            doc_id = f"bio_{b.system_id}"
            content = f"{b.name} ({b.branch}). Mechanisms: {' '.join(b.governing_mechanisms)}. Details: {b.molecular_and_biomechanical_details}"
            tokens = self._tokenize(content)
            self.documents[doc_id] = CitedDocument(
                doc_id=doc_id,
                citation_tag=f"[Citation: Biology/{b.system_id}]",
                domain="Biology",
                title=b.name,
                content=content,
                keywords=tokens
            )

        # 4. Philosophy
        for p in PhilosophyAndCreativityRegistry.get_all_philosophies():
            doc_id = f"phil_{p.name.lower().replace(' ', '_')}"
            content = f"{p.name} ({p.region_origin}). Thinkers: {', '.join(p.key_thinkers)}. Axiom: {p.core_axiom}. Application: {p.practical_application_for_ai}"
            tokens = self._tokenize(content)
            self.documents[doc_id] = CitedDocument(
                doc_id=doc_id,
                citation_tag=f"[Citation: Philosophy/{p.name}]",
                domain="Philosophy",
                title=p.name,
                content=content,
                keywords=tokens
            )

        # 5. Human Condition
        for h in HumanConditionRegistry.get_all():
            doc_id = f"human_{h.model_id}"
            content = f"{h.name}. Insight: {h.core_insight}. Vulnerability: {h.human_vulnerability}. Guidance: {h.companion_guidance}"
            tokens = self._tokenize(content)
            self.documents[doc_id] = CitedDocument(
                doc_id=doc_id,
                citation_tag=f"[Citation: HumanCondition/{h.model_id}]",
                domain="HumanCondition",
                title=h.name,
                content=content,
                keywords=tokens
            )

        # Build Inverted Word Index
        for doc_id, doc in self.documents.items():
            for token in doc.keywords:
                if token not in self.index:
                    self.index[token] = set()
                self.index[token].add(doc_id)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Performs TF-IDF style lexical retrieval with explicit source citations."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        doc_scores: Dict[str, float] = {}
        total_docs = len(self.documents)

        for token in query_tokens:
            matching_doc_ids = self.index.get(token, set())
            if not matching_doc_ids:
                continue

            # Inverse Document Frequency (IDF)
            idf = math.log((total_docs + 1) / (len(matching_doc_ids) + 1)) + 1.0

            for doc_id in matching_doc_ids:
                doc = self.documents[doc_id]
                # Term Frequency (TF)
                tf = 1.0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (tf * idf)

        # Sort by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            snippet = doc.content[:240] + ("..." if len(doc.content) > 240 else "")
            results.append(RetrievalResult(
                document=doc,
                score=round(score, 3),
                snippet=snippet
            ))

        return results

    def format_cited_context_block(self, query: str, top_k: int = 2) -> str:
        """Constructs grounded evidence block with citations for prompt injection."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        lines = ["VERIFIED LOCAL KNOWLEDGE CITATIONS:"]
        for r in results:
            lines.append(f"{r.document.citation_tag} {r.document.title}: {r.snippet}")
        return "\n".join(lines)
