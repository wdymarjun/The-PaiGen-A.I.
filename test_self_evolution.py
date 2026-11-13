"""
PaiGen AI: Safe Patch Proposal Demonstration (Safety-First Evolution)
Architect: Arjun Singh

Demonstrates PaiGen detecting an optimization opportunity and generating a
reviewable unified diff patch in `data/proposed_patches/` for developer review,
WITHOUT mutating or overwriting the active source code tree.
"""

from pathlib import Path
from paigen.core.evolution.decision_engine import EvolutionaryDecisionEngine, EvolutionTriggerReason
from paigen.core.evolution.meta_mutator import PatchProposalEngine
from paigen.core.emotion_tracker import EmotionTracker, EmotionFrame
from paigen.core.homeostasis import HomeostaticDrives


def run_safe_evolution_demo():
    repo_root = Path(__file__).parent.resolve()
    print("=" * 75)
    print("  PAIGEN A.I. SAFE PATCH PROPOSAL & ADAPTATION DEMO")
    print(f"  Architect: Arjun Singh | Repo: {repo_root.name}")
    print("=" * 75)

    tracker = EmotionTracker()
    drives = HomeostaticDrives()
    decision_engine = EvolutionaryDecisionEngine(tracker, drives)
    proposer = PatchProposalEngine(repo_root)

    print("\n[Step 1] Tracking Simulated Feedback...")
    directive = decision_engine.record_intervention_outcome(post_intervention_valence_delta=-0.45)
    print(f"  -> Should adapt? {directive.should_mutate}")
    print(f"  -> Reason: {directive.reason.value if directive.reason else 'None'}")
    print(f"  -> Recommendation: {directive.recommended_adjustment}")

    print("\n[Step 2] Generating Non-Destructive Reviewable Patch...")
    sample_proposed_code = '''# Proposed update for developer review
def decide_intervention_policy(smoothed_valence, arousal, fatigue, intrusiveness):
    if fatigue > 0.8:
        return "TRIGGER_GENTLE_REST_NUDGE"
    return "MAINTAIN_SUPPORTIVE_SILENCE"
'''
    success, message, patch_path = proposer.generate_reviewable_patch(
        target_relative_path="paigen/core/homeostasis.py",
        proposed_code=sample_proposed_code,
        rationale="Raise silence threshold in response to negative valence feedback"
    )

    print(f"  -> Success: {success}")
    print(f"  -> Message: {message}")
    if patch_path and patch_path.exists():
        print(f"  -> Patch Location: {patch_path.relative_to(repo_root)}")
        print("\n[Step 3] Preview of Proposed Patch (Unified Diff):")
        with open(patch_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[:12]:
                print(f"     {line.strip()}")

    print("\n" + "=" * 75)
    print("  Source tree remains 100% UNTOUCHED and intact.")
    print("  Developer has full review and approval control over proposed evolution.")
    print("=" * 75)


if __name__ == "__main__":
    run_safe_evolution_demo()
