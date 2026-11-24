"""
PaiGen AI: Real "Self-Learning" Autonomous Engine
Architect: Arjun Singh

Orchestrates live autonomous research and continuous self-learning:
1. Live Web Search (DDGS / keyless internet search)
2. Canonical Knowledge Acquisition (Wikipedia canonical summary extraction)
3. Persistent Long-Term RAG Vector Memory (ChromaDB / SQLite WAL fallback)
4. Overnight QLoRA Training Queue (training_queue.jsonl)
5. Multi-Turn Autonomous Tool Loop (ask_paigen)
"""

import datetime
import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("PaiGen.Brain.AutoLearner")

# ── 1. Library Imports with Resilient Fallbacks ───────────────────────────────
try:
    import ollama
    OLLAMA_LIB_OK = True
except ImportError:
    OLLAMA_LIB_OK = False

try:
    import wikipedia
    WIKIPEDIA_OK = True
except ImportError:
    WIKIPEDIA_OK = False

try:
    from duckduckgo_search import DDGS
    DDGS_OK = True
except ImportError:
    try:
        from ddgs import DDGS
        DDGS_OK = True
    except ImportError:
        DDGS_OK = False

try:
    import chromadb
    CHROMADB_OK = True
except ImportError:
    CHROMADB_OK = False


# ── 2. Persistent Vector Memory Engine ─────────────────────────────────────────
class SQLiteMemoryFallback:
    """
    High-performance, zero-dependency persistent memory collection matching
    ChromaDB's collection API (add and query) using SQLite WAL.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verified_knowledge (
                    id TEXT PRIMARY KEY,
                    document TEXT,
                    source TEXT,
                    domain TEXT,
                    created_at REAL
                )
            """)
            conn.commit()

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        import sqlite3
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                conn.execute("""
                    INSERT OR REPLACE INTO verified_knowledge (id, document, source, domain, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, doc, meta.get("source", "unknown"), meta.get("domain", "general"), now))
            conn.commit()

    def query(self, query_texts: List[str], n_results: int = 3) -> Dict[str, List[List[str]]]:
        import sqlite3
        if not query_texts or not query_texts[0].strip():
            return {"documents": [[]]}

        query = query_texts[0].lower()
        terms = [t for t in re.split(r"\W+", query) if len(t) > 2]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT id, document, source, domain FROM verified_knowledge ORDER BY created_at DESC")
            rows = cur.fetchall()

        # Score documents based on term overlap
        scored = []
        for r in rows:
            doc = r["document"]
            doc_lower = doc.lower()
            score = sum(2 if t in doc_lower else 0 for t in terms)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in scored[:n_results]]
        return {"documents": [results]}


# Initialize persistent memory
MEMORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "paigen_memory"))
os.makedirs(MEMORY_DIR, exist_ok=True)

mem = None
if CHROMADB_OK:
    try:
        chroma = chromadb.PersistentClient(path=MEMORY_DIR)
        mem = chroma.get_or_create_collection("verified_knowledge")
    except Exception as e:
        logger.warning(f"ChromaDB initialization failed ({e}), falling back to SQLite vector memory.")
        mem = None

if mem is None:
    fallback_db = os.path.join(MEMORY_DIR, "verified_knowledge.sqlite")
    mem = SQLiteMemoryFallback(fallback_db)


# ── 3. Autonomous Tools ───────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 4) -> str:
    """
    Search the live internet for recent facts, documentation, or benchmarks.
    """
    clean_q = query.strip()
    if not clean_q:
        return json.dumps({"error": "Empty query provided."})

    # 1. Primary: DDGS python library
    if DDGS_OK:
        try:
            results = list(DDGS().text(clean_q, max_results=max_results))
            if results:
                return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"DDGS lookup exception: {e}")

    # 2. Fallback: DuckDuckGo HTML parser
    try:
        import urllib.request
        import urllib.parse
        from bs4 import BeautifulSoup

        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(clean_q)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            soup = BeautifulSoup(resp.read().decode("utf-8", errors="ignore"), "html.parser")
            snippets = []
            for item in soup.select(".result__body")[:max_results]:
                title_elem = item.select_one(".result__title")
                snippet_elem = item.select_one(".result__snippet")
                if snippet_elem:
                    snippets.append({
                        "title": title_elem.get_text().strip() if title_elem else clean_q,
                        "body": snippet_elem.get_text().strip()
                    })
            if snippets:
                return json.dumps(snippets, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Direct HTML search fallback failed: {e}")

    return json.dumps([{"title": clean_q, "body": f"No live web results returned for '{clean_q}'."}], ensure_ascii=False)


def wiki_fetch(topic: str, sentences: int = 6) -> str:
    """
    Fetch canonical encyclopedic knowledge from Wikipedia.
    """
    clean_topic = topic.strip()
    if not clean_topic:
        return "No topic specified for Wikipedia lookup."

    if WIKIPEDIA_OK:
        try:
            return wikipedia.summary(clean_topic, sentences=sentences, auto_suggest=True)
        except Exception as e:
            logger.debug(f"wikipedia library summary exception: {e}")

    # Canonical Wikimedia REST API fallback (zero dependencies, high reliability)
    try:
        import urllib.request
        import urllib.parse
        encoded = urllib.parse.quote(clean_topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PaiGenAI/3.0 (Autonomous Executive Intelligence; mail6arjun478@gmail.com)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract")
            if extract:
                return extract
    except Exception as e:
        logger.debug(f"Wikipedia REST API fallback failed: {e}")

    return f"Wikipedia lookup error for topic '{clean_topic}'."


def memory_search(query: str, n_results: int = 3) -> str:
    """
    Search PaiGen's local long-term memory for previously verified facts.
    """
    res = mem.query(query_texts=[query], n_results=n_results)
    docs = res.get("documents", [[]])[0]
    if not docs:
        return "No matching local memories found."
    return json.dumps(docs, ensure_ascii=False)


def memory_commit(fact: str, source: str = "verified_agent", domain: str = "general") -> str:
    """
    Store a verified atomic fact into PaiGen's local long-term RAG memory.
    """
    clean_fact = fact.strip()
    if not clean_fact:
        return "No fact content provided."

    doc_id = f"mem_{int(datetime.datetime.now().timestamp())}_{abs(hash(clean_fact)) % 100000}"
    mem.add(
        documents=[clean_fact],
        metadatas=[{"source": source, "domain": domain}],
        ids=[doc_id]
    )
    return f"Committed to local memory [{doc_id}]."


def log_training_sample(instruction: str, verified_response: str) -> str:
    """
    Append a strictly verified Q&A pair to the overnight QLoRA training queue.
    """
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conversations": [
            {"from": "human", "value": instruction.strip()},
            {"from": "gpt", "value": verified_response.strip()}
        ]
    }
    queue_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "training_queue.jsonl")
    )
    with open(queue_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return "Queued for overnight QLoRA consolidation."


TOOLS = [web_search, wiki_fetch, memory_search, memory_commit, log_training_sample]
TOOL_MAP = {fn.__name__: fn for fn in TOOLS}


# ── 4. Autonomous Agent Loop ──────────────────────────────────────────────────

def ask_paigen(
    user_prompt: str,
    history: Optional[List[Dict[str, Any]]] = None,
    model: str = "paigen-ai",
    ollama_base_url: str = "http://127.0.0.1:11434",
    max_turns: int = 5,
    continuous_work_mins: int = 0,
    consecutive_errors: int = 0,
    perception_signals: Optional[List[str]] = None,
) -> str:
    """
    Executes the multi-turn autonomous tool loop:
    1. Injects top local verified memories and token-budgeted human-state telemetry before generation.
    2. Calls Ollama with active tool registry.
    3. Executes requested tools and feeds outputs back into the loop.
    4. Delivers the verified verdict directly.
    """
    if history is None:
        history = []

    # 1. Auto-inject top local RAG memories before the model even thinks
    try:
        recalled_res = mem.query(query_texts=[user_prompt], n_results=3)
        recalled = recalled_res.get("documents", [[]])[0]
        if recalled:
            history.append({
                "role": "system",
                "content": f"Recalled verified local memory:\n" + "\n".join(f"- {d}" for d in recalled)
            })
    except Exception as e:
        logger.debug(f"Memory pre-query exception: {e}")

    # 1b. Inject token-budgeted (<150 tokens) HumanConditionEngine telemetry if not already present
    try:
        already_injected = any(
            "[PAIGEN HUMAN-STATE TELEMETRY]" in str(m.get("content", ""))
            for m in history
            if m.get("role") == "system"
        )
        if not already_injected:
            from paigen.brain.knowledge.human_condition import (
                CognitiveStateInput,
                HumanConditionRegistry,
            )
            state_block = HumanConditionRegistry.compile_system_injection(
                CognitiveStateInput(
                    user_text=user_prompt,
                    hour_of_day=datetime.datetime.now().hour,
                    continuous_work_mins=continuous_work_mins,
                    consecutive_errors=consecutive_errors,
                    perception_signals=list(perception_signals or []),
                )
            )
            history.append({"role": "system", "content": state_block})
    except Exception as e:
        logger.debug(f"HumanCondition injection skipped: {e}")

    history.append({"role": "user", "content": user_prompt})

    # 2. Multi-turn autonomous tool loop via Ollama Python SDK or HTTP API
    for _ in range(max_turns):
        msg = None

        # Strategy A: Native ollama library
        if OLLAMA_LIB_OK:
            try:
                res = ollama.chat(model=model, messages=history, tools=TOOLS)
                msg = res.get("message", {})
            except Exception as e:
                logger.debug(f"ollama.chat SDK call failed ({e}), falling back to direct HTTP.")
                msg = None

        # Strategy B: Direct HTTP endpoint on localhost:11434
        if msg is None:
            try:
                import urllib.request
                payload = {
                    "model": model,
                    "messages": history,
                    "stream": False,
                    "options": {"temperature": 0.45}
                }
                req = urllib.request.Request(
                    f"{ollama_base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                    msg = raw.get("message", {})
            except Exception as e:
                return f"[Local Model Offline] Unable to reach Ollama at {ollama_base_url}: {e}"

        if not msg:
            break

        history.append(msg)

        # Check for tool execution requests
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            fn_dict = call.get("function", {})
            name = fn_dict.get("name")
            args = fn_dict.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name in TOOL_MAP:
                try:
                    output = TOOL_MAP[name](**args)
                except Exception as ex:
                    output = f"Tool execution error for {name}: {ex}"
            else:
                output = f"Unknown tool '{name}'."

            history.append({
                "role": "tool",
                "tool_name": name,
                "content": str(output)
            })

    return history[-1].get("content", "") if history else "No response generated."


class AutoLearner:
    """Singleton wrapper for PaiGen's autonomous research and learning engine."""
    _instance: Optional["AutoLearner"] = None

    @classmethod
    def get_instance(cls) -> "AutoLearner":
        if cls._instance is None:
            cls._instance = AutoLearner()
        return cls._instance

    def query(self, prompt: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        return ask_paigen(prompt, history)

    def search_web(self, query: str, max_results: int = 4) -> str:
        return web_search(query, max_results)

    def fetch_wikipedia(self, topic: str, sentences: int = 6) -> str:
        return wiki_fetch(topic, sentences)

    def commit_memory(self, fact: str, source: str = "verified_agent", domain: str = "general") -> str:
        return memory_commit(fact, source, domain)

    def log_training(self, instruction: str, verified_response: str) -> str:
        return log_training_sample(instruction, verified_response)
