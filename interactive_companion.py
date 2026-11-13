"""
PaiGen AI: Live Interactive Text Companion CLI
Architect: Arjun Singh

Dependable, privacy-first local text assistant.
Features:
- Live line-by-line token streaming (`query_stream`) as the default experience.
- Balanced user/assistant turn pair context history.
- Explicit override management: `/override <key> <val>`, `/unoverride <key>`, `/overrides`.
- Complete `/reset`: Clears all persistent overrides from SQLite and resets in-memory state.
- Grounded citations lookup: `/cite <topic>`.
"""

import sys
from datetime import datetime
from typing import Optional

from paigen.brain.cognitive import CognitiveBrain, TelemetryContext
from paigen.brain.interactive_attunement import HumanNatureAttuner
from paigen.brain.knowledge.retriever import LocalKnowledgeRetriever
from paigen.database.sqlite_store import SQLiteWALStore


def print_banner(db: SQLiteWALStore):
    overrides = db.list_overrides()
    forced_phase = overrides.get("circadian_phase")
    phase = HumanNatureAttuner.get_current_circadian_phase(forced_phase_name=forced_phase)

    print("=" * 75)
    print("  PAIGEN A.I. — DEPENDABLE LOCAL ASSISTANT")
    print("  Created & Architected by Arjun Singh | 100% Local & Privacy-First")
    print(f"  Local Clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Active Circadian Phase: {phase.phase_name}")
    print(f"  Active Overrides: {overrides if overrides else 'None (Heuristic Mode)'}")
    print("=" * 75)
    print("Commands:")
    print("  /history             -> View persistent message history from SQLite")
    print("  /clear               -> Securely clear current chat session history")
    print("  /cite <question>     -> Retrieve verified sources & literature citations")
    print("  /override <key> <v>  -> Set manual override (e.g. /override mood Focus)")
    print("  /unoverride <key>    -> Remove a specific manual override")
    print("  /overrides           -> View all active persistent overrides")
    print("  /reset               -> Reset in-memory telemetry AND wipe all overrides")
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
                history = db.get_session_turn_pairs(session_id, max_pairs=5)
                print(f"\n--- Context Turn Pairs ({len(history)} messages) ---")
                for msg in history:
                    print(f"  [{msg.role.upper()}]: {msg.content[:120]}...")
                print("---------------------------------------------------------")
                continue

            elif user_msg.lower() == "/overrides":
                all_o = db.list_overrides()
                print(f"[*] Active Overrides in SQLite: {all_o if all_o else 'None'}")
                continue

            elif user_msg.lower().startswith("/unoverride"):
                parts = user_msg.split(maxsplit=1)
                if len(parts) < 2:
                    print("[!] Usage: /unoverride <key>")
                    continue
                key = parts[1].strip()
                if db.delete_override(key):
                    print(f"[*] Removed override: '{key}'.")
                else:
                    print(f"[!] No active override found for: '{key}'.")
                continue

            elif user_msg.lower().startswith("/cite"):
                query = user_msg[5:].strip()
                if not query:
                    print("[!] Usage: /cite <topic or question>")
                    continue
                results = retriever.retrieve(query, top_k=2)
                if not results:
                    print(f"[*] No relevant verified sources found for: '{query}'")
                else:
                    print(f"\n--- Verified Citations for: '{query}' ---")
                    for r in results:
                        print(f"  Title: {r.document.title}")
                        print(f"  Source: {r.document.citation_tag}")
                        print(f"  Evidence: {r.snippet}\n")
                    print("-------------------------------------------------")
                continue

            elif user_msg.lower().startswith("/override"):
                parts = user_msg.split(maxsplit=2)
                if len(parts) < 3:
                    print("[!] Usage: /override <key> <value> (e.g. /override mood Focus)")
                    continue
                key, val = parts[1], parts[2]
                db.set_override(key, val)
                print(f"[*] Active override set: {key} = '{val}'. This directly drives persona & context.")
                continue

            elif user_msg.lower() == "/reset":
                # Wipe persistent overrides in SQLite
                wiped = db.clear_all_overrides()
                # Reset in-memory telemetry
                ctx.inferred_fatigue = 0.1
                ctx.inferred_valence = 0.2
                ctx.inferred_arousal = 0.5
                ctx.is_deep_focus = False
                ctx.is_user_override = False
                ctx.telemetry_confidence = 0.60
                print(f"[*] Complete reset: Wiped {wiped} persistent override(s) and returned telemetry to baseline.")
                continue

            # Stream response live to terminal as default experience
            print(f"\n[PaiGen AI] > ", end="", flush=True)
            for token in brain.query_stream(user_msg, ctx=ctx):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n[*] PaiGen: Session safely suspended. Take care, Arjun.")
            break


if __name__ == "__main__":
    run_interactive_session()
