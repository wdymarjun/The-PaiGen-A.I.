"""
PaiGen AI: Real Source-Backed Cited Knowledge Store & Retrieval Evaluation
Evaluates retrieval accuracy, real bibliographic literature references,
and handling when no relevant sources exist.
"""

import unittest
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever


class TestKnowledgeRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LocalKnowledgeRetriever()

    def test_french_revolution_retrieval_with_real_citations(self):
        query = "What were the fatal mistakes of the French Revolution and Robespierre?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "History")
        self.assertIn("french_revolution", top_doc.doc_id)
        # Verify real literature source
        self.assertIn("François Furet", top_doc.primary_source_author)
        self.assertIn("The French Revolution", top_doc.primary_source_work)
        self.assertIn("[Source:", top_doc.citation_tag)

    def test_ramanujan_mathematics_with_cambridge_citations(self):
        query = "Explain Ramanujan modular forms and partition formulas in number theory"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Mathematics")
        self.assertIn("ramanujan", top_doc.doc_id)
        self.assertIn("Cambridge University Press", top_doc.primary_source_work)
        self.assertIn("10.1017", top_doc.canonical_reference)

    def test_sliding_filament_biology_with_guyton_hall_citations(self):
        query = "How do muscles contract using actin and myosin sliding filaments?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Biology")
        self.assertIn("musculoskeletal", top_doc.doc_id)
        self.assertIn("Arthur C. Guyton", top_doc.primary_source_author)
        self.assertIn("Textbook of Medical Physiology", top_doc.primary_source_work)

    def test_stoicism_philosophy_with_marcus_aurelius_citations(self):
        query = "What is the dichotomy of control in Stoicism by Marcus Aurelius?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Philosophy")
        self.assertIn("stoicism", top_doc.doc_id)
        self.assertIn("Marcus Aurelius", top_doc.primary_source_author)
        self.assertIn("Meditations", top_doc.primary_source_work)

    def test_no_match_query_reports_no_sources(self):
        query = "xyzqwe1234 nonexistentterm9999 unindexedgibberish"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertEqual(len(results), 0)

        context_block = self.retriever.format_cited_context_block(query, top_k=2)
        self.assertIn("No relevant verified sources found", context_block)


if __name__ == "__main__":
    unittest.main()
