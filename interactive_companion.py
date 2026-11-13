"""
PaiGen AI: Live Interactive Text Companion CLI
Architect: Arjun Singh

Dependable, privacy-first local text assistant.
Features:
- Persistent SQLite WAL chat history with user controls (/history, /clear).
- Grounded Cited Knowledge Retrieval (/cite <topic>).
- Transparent state estimation with explicit user manual overrides (/override <key> <val>).
- 100% offline localhost execution.
"""

import sys
from datetime import datetime
from typing import Optional

from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.brain.interactive_attunement import HumanNatureAttuner
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever
from paigen.database.sqlite_store import SQLiteWALStore


def print_banner(db: SQLiteWALStore):
    phase = HumanNatureAttuner.get_current_circadian_phase()
    overrides = db.list_overrides()

    print("=" * 75)
    print("  PAIGEN A.I. — DEPENDABLE LOCAL ASSISTANT")
    print("  Created & Architected by Arjun Singh | 100% Local & Privacy-First")
    print(f"  Local Clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Inferred Circadian Phase: {phase.phase_name} (Confidence: 65% - Estimate)")
    if overrides:
        print(f"  Active User Overrides: {overrides}")
    print("=" * 75)
    print("Commands:")
    print("  /history             -> View persistent message history from SQLite")
    print("  /clear               -> Securely clear current chat session history")
    print("  /cite <question>     -> Retrieve verified sources & citations")
    print("  /override <key> <v>  -> Set manual override (e.g. /override mood Focused)")
    print("  /reset               -> Reset state to balanced baseline")
    print("  /exit                -> Exit session")
    print("-" * 75)


def run_interactive_session():
    db = SQLiteWALStore()
    retriever = LocalKnowledgeRetriever()
    brain = CognitiveBrain(db_store=db, retriever=retriever)

    print_banner(db)
    session_id = "default_session"
    db.get_or_create_session(session_id, title="Primary CLI Session")

    ctx = TelemetryContext(
        user_name="Arjun",
        session_id=session_id,
        inferred_valence=0.2,
        inferred_arousal=0.5,
        inferred_fatigue=0.1,
        is_deep_focus=False,
        trust_score=15.0,
        primary_language="bilingual"
    )

    while True:
        try:
            prompt_header = f"\n[Arjun (V:{ctx.inferred_valence:+.1f}, F:{ctx.inferred_fatigue:.1f})] > "
            user_msg = input(prompt_header).strip()
            if not user_msg:
                continue

            # Command Handling
            if user_msg.lower() == "/exit":
                print("\n[*] PaiGen: Alvida Arjun. Session safely persisted to local SQLite.")
                break

            elif user_msg.lower() == "/clear":
                deleted = db.clear_session_history(session_id)
                print(f"[*] Cleared {deleted} messages from current session history.")
                continue

            elif user_msg.lower() == "/history":
                history = db.get_session_history(session_id)
                print(f"\n--- Persistent Session History ({len(history)} messages) ---")
                for msg in history:
                    print(f"  [{msg.role.upper()}]: {msg.content[:120]}...")
                print("---------------------------------------------------------")
                continue

            elif user_msg.lower().startswith("/cite"):
                query = user_msg[5:].strip()
                if not query:
                    print("[!] Usage: /cite <topic or question>")
                    continue
                results = retriever.retrieve(query, top_k=2)
                print(f"\n--- Verified Local Citations for: '{query}' ---")
                for r in results:
                    print(f"  {r.document.citation_tag} (Score: {r.score})")
                    print(f"  Title: {r.document.title}")
                    print(f"  Evidence: {r.snippet}\n")
                print("-------------------------------------------------")
                continue

            elif user_msg.lower().startswith("/override"):
                parts = user_msg.split(maxsplit=2)
                if len(parts) < 3:
                    print("[!] Usage: /override <key> <value> (e.g., /override circadian_phase Midday)")
                    continue
                key, val = parts[1], parts[2]
                db.set_override(key, val)
                print(f"[*] Override applied: {key} = '{val}' (will override heuristics)")
                continue

            elif user_msg.lower() == "/reset":
                ctx.inferred_fatigue = 0.1
                ctx.inferred_valence = 0.2
                ctx.inferred_arousal = 0.5
                ctx.is_deep_focus = False
                print("[*] Telemetry reset to balanced baseline.")
                continue

            # Query the brain
            response = brain.query(user_msg, ctx=ctx)
            print(f"\n[PaiGen AI] > {response}")

        except (KeyboardInterrupt, EOFError):
            print("\n[*] PaiGen: Session safely suspended. Take care, Arjun.")
            break


if __name__ == "__main__":
    run_interactive_session()
