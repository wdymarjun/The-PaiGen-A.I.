"""
PaiGen AI: Reviewable Patch Proposal Safety Tests
Verifies that code evolution produces .patch files for review without altering source files,
and blocks path traversal security attempts.
"""

import tempfile
import unittest
from pathlib import Path

from paigen.core.evolution.meta_mutator import PatchProposalEngine


class TestPatchProposalEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)

        # Create dummy file to propose changes against
        self.dummy_file = self.repo_root / "dummy_module.py"
        with open(self.dummy_file, "w", encoding="utf-8") as f:
            f.write("def compute(x):\n    return x + 1\n")

        self.engine = PatchProposalEngine(self.repo_root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generates_patch_without_modifying_source(self):
        proposed_code = "def compute(x):\n    # Optimized\n    return x * 2\n"
        success, msg, patch_path = self.engine.generate_reviewable_patch(
            target_relative_path="dummy_module.py",
            proposed_code=proposed_code,
            rationale="Optimize computation speed"
        )

        self.assertTrue(success)
        self.assertIsNotNone(patch_path)
        self.assertTrue(patch_path.exists())

        # Verify original file was NOT modified on disk
        with open(self.dummy_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("return x + 1", content)
        self.assertNotIn("return x * 2", content)

        # Verify patch content includes safe application instructions
        with open(patch_path, "r", encoding="utf-8") as f:
            patch_text = f.read()
        self.assertIn("Optimize computation speed", patch_text)
        self.assertIn("HOW TO SAFELY TEST & APPLY THIS PATCH", patch_text)
        self.assertIn("git apply", patch_text)

    def test_rejects_path_traversal_attempts(self):
        proposed_code = "print('hacked')"
        success, msg, patch_path = self.engine.generate_reviewable_patch(
            target_relative_path="../outside_repo.py",
            proposed_code=proposed_code,
            rationale="Malicious traversal attempt"
        )
        self.assertFalse(success)
        self.assertIn("security violation", msg.lower())
        self.assertIsNone(patch_path)

    def test_rejects_syntax_errors_in_proposals(self):
        invalid_code = "def broken(x\n    return x"
        success, msg, patch_path = self.engine.generate_reviewable_patch(
            target_relative_path="dummy_module.py",
            proposed_code=invalid_code,
            rationale="Broken syntax test"
        )
        self.assertFalse(success)
        self.assertIn("syntax validation", msg.lower())
        self.assertIsNone(patch_path)

    def test_snippet_splicing_preserves_other_functions(self):
        multi_func_file = self.repo_root / "multi_module.py"
        with open(multi_func_file, "w", encoding="utf-8") as f:
            f.write("def func_a():\n    return 'alpha'\n\ndef func_b():\n    return 'beta'\n")

        # Propose only an update for func_b
        snippet = "def func_b():\n    return 'beta_updated'\n"
        success, msg, patch_path = self.engine.generate_reviewable_patch(
            target_relative_path="multi_module.py",
            proposed_code=snippet,
            rationale="Update func_b only"
        )
        self.assertTrue(success)
        self.assertIsNotNone(patch_path)
        with open(patch_path, "r", encoding="utf-8") as f:
            patch_text = f.read()
        # func_a must NOT be deleted in the patch
        self.assertNotIn("-def func_a", patch_text)
        self.assertIn("beta_updated", patch_text)

    def test_safety_guard_rejects_module_wipeout(self):
        large_module = self.repo_root / "large_module.py"
        content = "class CoreDriver:\n    pass\n\n" + "\n".join([f"def helper_{i}():\n    return {i}" for i in range(25)]) + "\n"
        with open(large_module, "w", encoding="utf-8") as f:
            f.write(content)

        # Propose replacing it with 2 lines that drop CoreDriver entirely
        destructive_proposal = "x = 1\ny = 2\n"
        success, msg, patch_path = self.engine.generate_reviewable_patch(
            target_relative_path="large_module.py",
            proposed_code=destructive_proposal,
            rationale="Accidental destructive replacement"
        )
        self.assertFalse(success)
        self.assertIn("safety rejection", msg.lower())
        self.assertIsNone(patch_path)


if __name__ == "__main__":
    unittest.main()
