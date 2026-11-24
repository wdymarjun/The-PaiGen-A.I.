"""
Unit tests for PaiGen AI AutoLearner & Self-Learning Engine.
Architect: Arjun Singh
"""

import json
import os
import time
import unittest
from unittest.mock import patch, MagicMock

from paigen.brain.autolearner import (
    AutoLearner,
    web_search,
    wiki_fetch,
    memory_commit,
    memory_search,
    log_training_sample,
    ask_paigen,
    TOOLS,
    TOOL_MAP
)


class TestAutoLearner(unittest.TestCase):
    def test_tools_registry(self):
        self.assertIn("web_search", TOOL_MAP)
        self.assertIn("wiki_fetch", TOOL_MAP)
        self.assertIn("memory_search", TOOL_MAP)
        self.assertIn("memory_commit", TOOL_MAP)
        self.assertIn("log_training_sample", TOOL_MAP)
        self.assertEqual(len(TOOLS), 5)

    def test_memory_commit_and_search(self):
        test_fact = f"Ramanujan partition formula asymptote test fact {int(time.time())}"
        commit_res = memory_commit(test_fact, source="unit_test", domain="pure_math")
        self.assertIn("Committed to local memory", commit_res)

        search_res = memory_search("Ramanujan partition")
        self.assertIsInstance(search_res, str)
        self.assertIn("Ramanujan", search_res)

    def test_log_training_sample(self):
        instr = "Explain the French Revolution Reign of Terror failure mode."
        resp = "Ideological purism and economic collapse under fiat Assignats led to Thermidorian Reaction."
        res = log_training_sample(instr, resp)
        self.assertIn("Queued for overnight QLoRA consolidation", res)

        # Verify training_queue.jsonl exists and contains this entry
        queue_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "training_queue.jsonl")
        )
        self.assertTrue(os.path.exists(queue_file))
        with open(queue_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertTrue(len(lines) > 0)
        last_rec = json.loads(lines[-1])
        self.assertIn("conversations", last_rec)
        self.assertEqual(last_rec["conversations"][0]["value"], instr)
        self.assertEqual(last_rec["conversations"][1]["value"], resp)

    def test_wiki_fetch_fallback_or_live(self):
        summary = wiki_fetch("Python (programming language)", sentences=2)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)

    def test_web_search_structure(self):
        res = web_search("Ollama", max_results=2)
        self.assertIsInstance(res, str)
        parsed = json.loads(res)
        self.assertTrue(isinstance(parsed, (list, dict)))

    def test_singleton_instance(self):
        learner = AutoLearner.get_instance()
        self.assertIsNotNone(learner)
        self.assertEqual(learner, AutoLearner.get_instance())


if __name__ == "__main__":
    unittest.main()
