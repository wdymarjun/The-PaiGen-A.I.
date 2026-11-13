"""
PaiGen AI: MetaMutator Self-Rewriting Engine
Architect: Arjun Singh

Enables PaiGen AI to autonomously rewrite its own core source code based on
emotional feedback decisions. Includes AST validation, safety snapshots,
compilation harnesses, and dynamic hot-reloading in memory.
"""

import ast
import importlib
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from paigen.core.evolution.safety_harness import SafetyHarness
from paigen.core.evolution.decision_engine import EvolutionDirective, EvolutionTriggerReason

logger = logging.getLogger("PaiGen.Core.Evolution.MetaMutator")


class MetaMutator:
    """Executes safe, self-directed code transformations inside PaiGen's core."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.safety = SafetyHarness(repo_root)

    def apply_directive(self, directive: EvolutionDirective) -> Tuple[bool, str]:
        """Processes an evolution directive and mutates the target module."""
        if not directive.should_mutate:
            return False, "Directive indicated no mutation needed."

        target_file = self.repo_root / directive.target_module
        if not target_file.exists():
            return False, f"Target file does not exist: {target_file}"

        # 1. Create safety snapshot before altering anything
        snapshot_path = self.safety.create_snapshot(target_file)

        try:
            # 2. Read existing source code
            with open(target_file, "r", encoding="utf-8") as f:
                original_code = f.read()

            # 3. Generate evolved code for the targeted function
            mutated_code = self._synthesize_evolution(original_code, directive)

            if mutated_code == original_code:
                return False, "Synthesized code is identical to original; no mutation applied."

            # 4. Rigorous Safety Validation (AST + Compile Test)
            is_valid, err_msg = self.safety.validate_python_code(mutated_code)
            if not is_valid:
                logger.error(f"[X] Mutation failed safety validation: {err_msg}")
                self.safety.rollback(target_file, snapshot_path)
                return False, f"Safety check rejected mutation: {err_msg}"

            # 5. Write validated code directly into core
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(mutated_code)

            # 6. Dynamic In-Memory Hot-Reload
            module_name = directive.target_module.replace(".py", "").replace("/", ".").replace("\\", ".")
            try:
                mod = importlib.import_module(module_name)
                importlib.reload(mod)
                logger.info(f"[+] Successfully hot-reloaded evolved module: {module_name}")
            except Exception as reload_err:
                logger.error(f"[!] Hot reload failed: {reload_err}. Reverting snapshot.")
                self.safety.rollback(target_file, snapshot_path)
                return False, f"Runtime reload failed: {reload_err}"

            logger.info(f"[***] Code self-evolution successful for {directive.target_function} ({directive.reason})")
            return True, f"Successfully evolved {directive.target_function} in {target_file.name}"

        except Exception as e:
            logger.error(f"[X] Unexpected error during code evolution: {e}")
            self.safety.rollback(target_file, snapshot_path)
            return False, f"Exception during evolution: {str(e)}"

    def _synthesize_evolution(self, original_code: str, directive: EvolutionDirective) -> str:
        """
        Synthesizes improved AST / code block.
        Applies domain-specific evolutionary adaptation rules based on the trigger.
        """
        if directive.reason == EvolutionTriggerReason.INTRUSIVENESS_PENALTY:
            # User became agitated/frowning from too much interruption.
            # Evolve decide_intervention_policy to raise silence threshold:
            evolved_function = '''def decide_intervention_policy(
    smoothed_valence: float,
    arousal: float,
    fatigue: float,
    intrusiveness: float
) -> str:
    """
    [EVOLVED BY PAIGEN SELF-MUTATOR]
    Adaptive Update: User signaled distress from high intrusiveness.
    Raised intervention barrier and added empathetic restraint buffer.
    """
    if fatigue > 0.8:
        return "TRIGGER_GENTLE_REST_NUDGE"

    if smoothed_valence < -0.6:
        # User is deeply frustrated: Maintain silence unless explicitly asked
        return "MAINTAIN_SUPPORTIVE_SILENCE"

    if smoothed_valence > 0.4 and arousal > 0.6:
        return "OFFER_TACTICAL_COLLABORATION"

    return "OBSERVE_PASSIVELY"'''

            # Replace the old function definition cleanly using regex
            pattern = r"def decide_intervention_policy\(.*?\) -> str:.*?(?=\ndef |\Z)"
            mutated = re.sub(pattern, evolved_function, original_code, flags=re.DOTALL)
            return mutated

        elif directive.reason == EvolutionTriggerReason.TRUST_LEVEL_UP:
            # Trust unlocked: Allow higher temperature and witty banter
            evolved_function = '''def calculate_dynamic_temperature(
    valence: float,
    trust: float,
    is_work_session: bool
) -> float:
    """
    [EVOLVED BY PAIGEN SELF-MUTATOR]
    Adaptive Update: High trust unlocked. Increased creative dynamic range.
    """
    if is_work_session:
        return 0.30  # Laser precision during execution

    if trust >= 50.0:
        return 0.80  # Unlocked spontaneous creative banter

    return 0.55'''

            pattern = r"def calculate_dynamic_temperature\(.*?\) -> float:.*?(?=\ndef |\Z)"
            mutated = re.sub(pattern, evolved_function, original_code, flags=re.DOTALL)
            return mutated

        return original_code
