"""
PaiGen AI: Local Knowledge Store & Retrieval Evaluation
Evaluates retrieval accuracy and citation verification across a fixed suite of questions.
"""

import unittest
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever


class TestKnowledgeRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LocalKnowledgeRetriever()

    def test_french_revolution_retrieval(self):
        query = "What were the fatal mistakes of the French Revolution and Robespierre?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "History")
        self.assertIn("french_revolution", top_doc.doc_id)
        self.assertIn("[Citation: WorldHistory/french_revolution_1789]", top_doc.citation_tag)
        self.assertIn("Robespierre", top_doc.content)

    def test_ramanujan_mathematics_retrieval(self):
        query = "Explain Ramanujan modular forms and partition formulas in number theory"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Mathematics")
        self.assertIn("ramanujan", top_doc.doc_id)
        self.assertIn("Ramanujan", top_doc.content)
        self.assertIn("Partition", top_doc.content)

    def test_sliding_filament_biology_retrieval(self):
        query = "How do muscles contract using actin and myosin sliding filaments?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Biology")
        self.assertIn("musculoskeletal", top_doc.doc_id)
        self.assertIn("Troponin", top_doc.content)

    def test_stoicism_philosophy_retrieval(self):
        query = "What is the dichotomy of control in Stoicism by Marcus Aurelius?"
        results = self.retriever.retrieve(query, top_k=2)
        self.assertGreater(len(results), 0)

        top_doc = results[0].document
        self.assertEqual(top_doc.domain, "Philosophy")
        self.assertIn("stoicism", top_doc.doc_id)
        self.assertIn("Marcus Aurelius", top_doc.content)

    def test_cited_context_block_formatting(self):
        query = "Chanakya Arthashastra statecraft"
        context_block = self.retriever.format_cited_context_block(query, top_k=1)
        self.assertIn("VERIFIED LOCAL KNOWLEDGE CITATIONS:", context_block)
        self.assertIn("[Citation:", context_block)


if __name__ == "__main__":
    unittest.main()
