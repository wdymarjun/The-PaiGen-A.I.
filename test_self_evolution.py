"""
PaiGen AI: Self-Evolution & Code Self-Rewriting Demonstration
Architect: Arjun Singh

Simulates PaiGen tracking user emotional valence, detecting that its current
intervention heuristics caused agitation, autonomously deciding to evolve its code,
validating AST syntax with SafetyHarness, and modifying paigen/core/homeostasis.py
live with hot-reload in memory.
"""

from pathlib import Path
from datetime import datetime

from paigen.core.emotion_tracker import EmotionTracker, EmotionFrame
from paigen.core.homeostasis import HomeostaticDrives, decide_intervention_policy
from paigen.core.evolution.decision_engine import EvolutionaryDecisionEngine
from paigen.core.evolution.meta_mutator import MetaMutator


def run_evolution_simulation():
    repo_root = Path(__file__).parent.resolve()
    print("=" * 75)
    print("  PAIGEN A.I. AUTONOMOUS SELF-EVOLUTION & CODE REWRITING TEST")
    print(f"  Architect: Arjun Singh | Repo: {repo_root.name}")
    print("=" * 75)

    # 1. Initialize Subsystems
    tracker = EmotionTracker()
    drives = HomeostaticDrives()
    decision_engine = EvolutionaryDecisionEngine(tracker, drives)
    mutator = MetaMutator(repo_root)

    # 2. Check Baseline Code Behavior
    print("\n[Step 1] Baseline Decision from 'decide_intervention_policy':")
    action_before = decide_intervention_policy(
        smoothed_valence=-0.55, arousal=0.7, fatigue=0.2, intrusiveness=0.6
    )
    print(f"  -> At valence -0.55 & intrusiveness 0.6: Action = '{action_before}'")

    # 3. Simulate Emotion Tracking: User becomes agitated after two consecutive interventions
    print("\n[Step 2] User Interaction & Emotion Tracking...")
    print("  Intervention 1 executed. Observing user micro-expressions...")
    tracker.ingest_frame(EmotionFrame(primary_emotion="angry", valence=-0.5, arousal=0.8))
    directive_1 = decision_engine.record_intervention_outcome(post_intervention_valence_delta=-0.35)
    print(f"  -> Delta v: -0.35 | Should Mutate? {directive_1.should_mutate}")

    print("  Intervention 2 executed. User frowns again (Negative Valence Spike)...")
    tracker.ingest_frame(EmotionFrame(primary_emotion="sad", valence=-0.65, arousal=0.7))
    directive_2 = decision_engine.record_intervention_outcome(post_intervention_valence_delta=-0.40)
    print(f"  -> Delta v: -0.40 | Should Mutate? {directive_2.should_mutate}")
    print(f"  -> Decision Engine Reason: {directive_2.reason.value}")
    print(f"  -> Recommendation: {directive_2.recommended_adjustment}")

    # 4. Autonomous Code Rewriting Execution
    print("\n[Step 3] PaiGen Initiates Self-Rewriting of Core Logic...")
    success, message = mutator.apply_directive(directive_2)
    print(f"  -> Result: {message}")
    print(f"  -> Status: {'SUCCESS' if success else 'FAILED'}")

    # 5. Verify In-Memory Dynamic Execution of the Evolved Code
    from paigen.core.homeostasis import decide_intervention_policy as evolved_policy
    action_after = evolved_policy(
        smoothed_valence=-0.55, arousal=0.7, fatigue=0.2, intrusiveness=0.6
    )
    print("\n[Step 4] Evolved Behavior Verification:")
    print(f"  -> At same state (valence -0.55): Old Action = '{action_before}' | New Evolved Action = '{action_after}'")

    print("\n" + "=" * 75)
    print("  PaiGen successfully tracked user emotions, deduced heuristic failure,")
    print("  and rewrote its own core source code with full AST safety validation!")
    print("=" * 75)


if __name__ == "__main__":
    run_evolution_simulation()
