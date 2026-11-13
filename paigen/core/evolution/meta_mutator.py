"""
PaiGen AI: Reviewable Patch Proposal Engine
Architect: Arjun Singh

Safety-first evolution: Instead of overwriting source code in-place during runtime,
PaiGen proposes unified diff patches stored in `data/proposed_patches/` for human review,
automated unit test verification, and explicit developer approval.
"""

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
        self.repo_root = repo_root
        self.patch_dir = repo_root / "data" / "proposed_patches"
        self.patch_dir.mkdir(parents=True, exist_ok=True)
        self.safety = SafetyHarness(repo_root)

    def generate_reviewable_patch(
        self,
        target_relative_path: str,
        proposed_code: str,
        rationale: str
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Validates proposed Python code for syntax and generates a .patch file
        for developer review without modifying the active source tree.
        """
        target_file = self.repo_root / target_relative_path
        if not target_file.exists():
            return False, f"Target file does not exist: {target_file}", None

        # 1. Validate syntax of proposed code
        is_valid, err_msg = self.safety.validate_python_code(proposed_code)
        if not is_valid:
            return False, f"Proposed code failed syntax validation: {err_msg}", None

        # 2. Read original source code
        with open(target_file, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        proposed_lines = proposed_code.splitlines(keepends=True)

        # 3. Generate unified diff
        diff_lines = list(difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{target_relative_path}",
            tofile=f"b/{target_relative_path}",
            lineterm=""
        ))

        if not diff_lines:
            return False, "Proposed code is identical to original; no patch generated.", None

        # 4. Save patch file with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patch_filename = f"proposal_{Path(target_relative_path).stem}_{timestamp}.patch"
        patch_path = self.patch_dir / patch_filename

        header = [
            f"# PaiGen AI Proposed Code Modification\n",
            f"# Rationale: {rationale}\n",
            f"# Created: {datetime.now().isoformat()}\n",
            f"# Status: PENDING_HUMAN_REVIEW\n\n"
        ]

        with open(patch_path, "w", encoding="utf-8") as f:
            f.writelines(header)
            f.write("\n".join(diff_lines) + "\n")

        logger.info(f"[*] Generated reviewable patch: {patch_path.name}")
        return True, f"Patch generated successfully for human review at: {patch_path.name}", patch_path
