"""
PaiGen AI: Live Interactive Human Nature Companion CLI
Architect: Arjun Singh

Interactive terminal session with PaiGen AI.
Simulates real-time telemetry (valence, fatigue, time of day, trust score),
showcasing dynamic dual-persona adaptation and organic human nature attunement.
"""

import sys
from datetime import datetime
from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.brain.interactive_attunement import HumanNatureAttuner


def print_banner():
    phase = HumanNatureAttuner.get_current_circadian_phase()
    print("=" * 75)
    print("  PAIGEN A.I. — HUMAN NATURE INTERACTIVE COMPANION")
    print("  Created & Architected by Arjun Singh | 100% Local & Privacy-First")
    print(f"  Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Circadian Phase: {phase.phase_name}")
    print(f"  Biological State: {phase.biological_state}")
    print("=" * 75)
    print("Special Commands:")
    print("  /tired   -> Simulate physical fatigue & high adenosine")
    print("  /stress  -> Simulate compiler error & negative valence")
    print("  /focus   -> Simulate deep cognitive flow state")
    print("  /eureka  -> Simulate architectural breakthrough & high dopamine")
    print("  /reset   -> Return to balanced baseline")
    print("  /exit    -> End session")
    print("-" * 75)


def run_interactive_session():
    print_banner()
    brain = CognitiveBrain()
    ctx = TelemetryContext(
        user_name="Arjun",
        valence=0.2,
        arousal=0.5,
        fatigue_score=0.1,
        is_deep_focus=False,
        trust_score=15.0,
        primary_language="bilingual"
    )

    while True:
        try:
            user_msg = input(f"\n[Arjun (V:{ctx.valence:+.1f}, F:{ctx.fatigue_score:.1f})] > ").strip()
            if not user_msg:
                continue

            if user_msg.lower() == "/exit":
                print("\n[*] PaiGen: Alvida Arjun, take care of yourself. Systems in standby.")
                break
            elif user_msg.lower() == "/tired":
                ctx.fatigue_score = 0.85
                ctx.valence = -0.3
                ctx.arousal = 0.2
                ctx.is_deep_focus = False
                print("[!] [Simulated Telemetry]: High fatigue detected (Eyelid droop, adenosine saturation).")
                continue
            elif user_msg.lower() == "/stress":
                ctx.fatigue_score = 0.5
                ctx.valence = -0.7
                ctx.arousal = 0.85
                ctx.is_deep_focus = False
                print("[!] [Simulated Telemetry]: High stress / frustration detected (AU4 brow furrow, AU24 lip press).")
                continue
            elif user_msg.lower() == "/focus":
                ctx.fatigue_score = 0.1
                ctx.valence = 0.5
                ctx.arousal = 0.6
                ctx.is_deep_focus = True
                print("[!] [Simulated Telemetry]: Deep cognitive flow state locked in (AU4, steady gaze).")
                continue
            elif user_msg.lower() == "/eureka":
                ctx.fatigue_score = 0.05
                ctx.valence = 0.95
                ctx.arousal = 0.90
                ctx.is_deep_focus = False
                ctx.trust_score = min(100.0, ctx.trust_score + 5.0)
                print("[!] [Simulated Telemetry]: Eureka epiphany detected (AU1+AU2 brow rise, AU6 Duchenne smile).")
                continue
            elif user_msg.lower() == "/reset":
                ctx.fatigue_score = 0.1
                ctx.valence = 0.2
                ctx.arousal = 0.5
                ctx.is_deep_focus = False
                print("[*] [Simulated Telemetry]: Reset to balanced baseline.")
                continue

            # Query the brain
            response = brain.query(user_msg, ctx=ctx)
            print(f"\n[PaiGen AI] > {response}")

        except (KeyboardInterrupt, EOFError):
            print("\n[*] PaiGen: Session safely suspended. Take care, Arjun.")
            break


if __name__ == "__main__":
    run_interactive_session()
