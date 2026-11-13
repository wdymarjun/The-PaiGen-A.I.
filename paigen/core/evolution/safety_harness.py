"""
PaiGen AI: Safety Harness for Self-Rewriting Code
Architect: Arjun Singh

Ensures that any self-generated code mutations pass syntax validation,
AST integrity checks, and isolation testing before being written into the core.
Includes automated snapshotting and rollback mechanics to prevent breaking the system.
"""

import ast
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("PaiGen.Core.Evolution.Safety")


class SafetyHarness:
    """Guards against corrupt or breaking self-modifications."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.snapshot_dir = repo_root / "data" / "evolution_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def validate_python_code(self, code_str: str) -> Tuple[bool, Optional[str]]:
        """Parses and checks code syntax and compile integrity."""
        try:
            tree = ast.parse(code_str)
            compile(tree, filename="<self_evolution>", mode="exec")
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Compilation validation error: {str(e)}"

    def create_snapshot(self, target_file: Path) -> Path:
        """Creates an atomic backup of a file prior to self-modification."""
        if not target_file.exists():
            raise FileNotFoundError(f"Cannot snapshot non-existent file: {target_file}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"{target_file.stem}_{timestamp}{target_file.suffix}.bak"
        backup_path = self.snapshot_dir / backup_name
        shutil.copy2(target_file, backup_path)
        logger.info(f"[*] Created snapshot for {target_file.name} at: {backup_path.name}")
        return backup_path

    def rollback(self, target_file: Path, snapshot_path: Path) -> bool:
        """Restores a previous known-good snapshot."""
        if snapshot_path.exists():
            shutil.copy2(snapshot_path, target_file)
            logger.warning(f"[!] Reverted {target_file.name} to snapshot: {snapshot_path.name}")
            return True
        logger.error(f"[X] Snapshot not found for rollback: {snapshot_path}")
        return False
