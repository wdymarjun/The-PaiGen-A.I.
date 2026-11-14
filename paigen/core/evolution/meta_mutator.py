"""
PaiGen AI: Reviewable Patch Proposal Engine
Architect: Arjun Singh

Safety-first evolution: Instead of overwriting source code in-place during runtime,
PaiGen proposes unified diff patches stored in `data/proposed_patches/` for human review,
automated unit test verification, and explicit developer approval.
Prevents path traversal and validates targets stay strictly inside repo bounds.
"""

import ast
import difflib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from paigen.core.evolution.safety_harness import SafetyHarness
from paigen.core.evolution.decision_engine import EvolutionDirective

logger = logging.getLogger("PaiGen.Core.Evolution.PatchProposer")


class PatchProposalEngine:
    """Generates reviewable patches without modifying running runtime source code."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self.patch_dir = self.repo_root / "data" / "proposed_patches"
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self.safety = SafetyHarness(self.repo_root)

    def generate_reviewable_patch(
        self,
        target_relative_path: str,
        proposed_code: str,
        rationale: str
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Validates target stays inside repository bounds, checks syntax via AST,
        and generates a safe .patch file for manual developer review.
        """
        # Security: Prevent path traversal outside repository
        target_file = (self.repo_root / target_relative_path).resolve()
        try:
            target_file.relative_to(self.repo_root)
        except ValueError:
            return False, f"Security Violation: Target path '{target_relative_path}' resolves outside repo boundary.", None

        if not target_file.exists():
            return False, f"Target file does not exist: {target_relative_path}", None

        # 1. Validate syntax of proposed code
        is_valid, err_msg = self.safety.validate_python_code(proposed_code)
        if not is_valid:
            return False, f"Proposed code failed syntax validation: {err_msg}", None

        # 2. Read original source code
        with open(target_file, "r", encoding="utf-8") as f:
            original_lines = [line.rstrip("\r\n") for line in f]

        # 3. Intelligent AST-based Snippet Splicing and Safety Guards
        # Prevents small function snippets from accidentally deleting entire modules
        try:
            tree_orig = ast.parse("\n".join(original_lines))
            tree_prop = ast.parse(proposed_code)

            orig_defs = {n.name: n for n in tree_orig.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            prop_defs = {n.name: n for n in tree_prop.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

            # Case A: Proposed code is a targeted function/class snippet
            if len(prop_defs) > 0 and len(prop_defs) < len(orig_defs):
                spliced_lines = list(original_lines)
                matching_nodes = sorted(
                    [n for n in tree_orig.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in prop_defs],
                    key=lambda x: x.lineno,
                    reverse=True
                )
                for node in matching_nodes:
                    snip_node = prop_defs[node.name]
                    snip_lines = proposed_code.splitlines()[snip_node.lineno - 1 : snip_node.end_lineno]
                    spliced_lines = spliced_lines[:node.lineno - 1] + snip_lines + spliced_lines[node.end_lineno:]
                proposed_lines = spliced_lines
            else:
                proposed_lines = proposed_code.splitlines()
                # Case B: Safety guard against accidental full-module wipeout
                if len(original_lines) > 20 and len(proposed_lines) < len(original_lines) * 0.6:
                    missing_classes = set(orig_defs.keys()) - set(prop_defs.keys())
                    if missing_classes:
                        return False, f"Safety Rejection: Proposed patch drops critical definitions {missing_classes} and deletes >40% of target file.", None
        except Exception:
            proposed_lines = proposed_code.splitlines()

        # 4. Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{target_relative_path}",
            tofile=f"b/{target_relative_path}",
            lineterm=""
        ))

        if not diff_lines:
            return False, "Proposed code is identical to original; no patch generated.", None

        # 4. Save patch file with metadata in Git mailbox format compatible with git apply
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patch_filename = f"proposal_{Path(target_relative_path).stem}_{timestamp}.patch"
        patch_path = self.patch_dir / patch_filename

        header = (
            f"From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
            f"From: PaiGen AI <paigen@local>\n"
            f"Subject: [PATCH] {rationale}\n\n"
            f"PaiGen AI Proposed Code Modification (PENDING HUMAN REVIEW)\n"
            f"Target File: {target_relative_path}\n"
            f"Rationale: {rationale}\n"
            f"Created: {datetime.now().isoformat()}\n\n"
            f"HOW TO SAFELY TEST & APPLY THIS PATCH:\n"
            f"1. Review the diff below carefully for behavioral correctness.\n"
            f"2. Test patch dry-run: git apply --check {patch_path.relative_to(self.repo_root)}\n"
            f"3. Run regression suite: python -m unittest discover tests\n"
            f"4. Apply if satisfied: git apply {patch_path.relative_to(self.repo_root)}\n\n"
            f"---\n"
        )

        with open(patch_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.write("\n".join(diff_lines) + "\n")

        logger.info(f"[*] Generated reviewable patch: {patch_path.name}")
        return True, f"Patch generated successfully for human review at: {patch_path.name}", patch_path
