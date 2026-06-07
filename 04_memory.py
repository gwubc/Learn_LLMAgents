"""
LLM Agent Memory System with Importance & Decay
================================================
Each memory has an importance score (0-1) set at creation.
Effective importance decays exponentially with time:

    effective = importance * e^(-decay_rate * days_old)

Retrieval ranks by:  cosine_similarity(query, memory) * effective_importance

Embeddings are produced by Qwen3-Embedding-8B via ModelScope and cached
on each Memory object so content is never re-embedded.
"""

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── tunables ─────────────────────────────────────────────────────────────────
DECAY_RATE = 0.5      # importance halves every ~1.4 days
ACCESS_BOOST = 0.04   # per recall; capped at +0.20
EMB_MODEL = "Qwen/Qwen3-Embedding-8B"


# ── embedding client ──────────────────────────────────────────────────────────

class Embedder:
    """Thin wrapper around ModelScope embeddings with an in-process cache."""

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("MODELSCOPE_API_KEY"),
            base_url="https://api-inference.modelscope.ai/v1",
        )
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            resp = self._client.embeddings.create(
                model=EMB_MODEL,
                input=text,
                encoding_format="float",
            )
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            vec /= np.linalg.norm(vec) + 1e-10   # unit-normalise once
            self._cache[text] = vec
        return self._cache[text]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))   # both already unit-normalised


# ── memory primitives ────────────────────────────────────────────────────────

@dataclass
class Memory:
    content: str
    importance: float              # 0.0–1.0, set at creation
    created_at: datetime
    embedding: np.ndarray = field(repr=False, default=None)   # set by MemoryStore
    last_accessed: datetime = field(init=False)
    access_count: int = 0

    def __post_init__(self):
        self.last_accessed = self.created_at

    def effective_importance(self, now: datetime) -> float:
        """Decayed importance; accessing a memory partially refreshes it."""
        days_old = (now - self.created_at).total_seconds() / 86400
        boost = min(0.20, self.access_count * ACCESS_BOOST)
        return min(1.0, (self.importance + boost) * math.exp(-DECAY_RATE * days_old))


class MemoryStore:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._memories: list[Memory] = []

    def add(self, content: str, importance: float,
            now: Optional[datetime] = None) -> Memory:
        now = now or datetime.now()
        mem = Memory(
            content=content,
            importance=max(0.0, min(1.0, importance)),
            created_at=now,
            embedding=self._embedder.embed(content),
        )
        self._memories.append(mem)
        return mem

    def recall(self, query: str, top_k: int = 3,
               now: Optional[datetime] = None) -> list[Memory]:
        now = now or datetime.now()
        q_vec = self._embedder.embed(query)

        def score(m: Memory) -> float:
            sim = max(0.0, Embedder.cosine(q_vec, m.embedding))  # clamp negatives to 0
            return sim * m.effective_importance(now)

        ranked = sorted(self._memories, key=score, reverse=True)
        results = ranked[:top_k]
        for m in results:
            m.last_accessed = now
            m.access_count += 1
        return results

    def snapshot(self, now: Optional[datetime] = None) -> list[tuple[float, Memory]]:
        now = now or datetime.now()
        return sorted(
            [(m.effective_importance(now), m) for m in self._memories],
            key=lambda x: x[0],
            reverse=True,
        )


# ── agent ────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": (
                "Save an important fact to long-term memory. "
                "Set importance (0.0–1.0): 1.0 = critical, 0.5 = useful, 0.1 = trivial."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact or information to remember.",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score between 0.0 and 1.0.",
                    },
                },
                "required": ["content", "importance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": (
                "Search long-term memory for facts relevant to a query. "
                "Returns the most important and recent matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords describing what you want to remember.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max number of memories to return (default 3).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a personal AI assistant with a memory system.

Rules:
- Before answering any question about facts, people, or preferences: call recall_memories.
- After learning something new and important: call store_memory with an appropriate importance score.
- Importance guide: 1.0 = critical/urgent, 0.7 = very useful, 0.5 = somewhat useful, 0.2 = trivial.
- Be concise. If memories are relevant, cite them in your answer."""


class MemoryAgent:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_URL"),
        )
        self.model = os.getenv("LLM_MODEL")
        self.now = datetime.now()   # can be overridden to simulate time travel

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "store_memory":
            mem = self.store.add(
                content=args["content"],
                importance=args["importance"],
                now=self.now,
            )
            return (
                f"Stored: '{mem.content}' "
                f"(importance={mem.importance:.2f}, "
                f"created={mem.created_at.strftime('%Y-%m-%d %H:%M')})"
            )
        elif name == "recall_memories":
            memories = self.store.recall(
                query=args["query"],
                top_k=args.get("top_k", 3),
                now=self.now,
            )
            if not memories:
                return "No relevant memories found."
            q_vec = self.store._embedder.embed(args["query"])
            rows = []
            for m in memories:
                sim = max(0.0, Embedder.cosine(q_vec, m.embedding))
                eff = m.effective_importance(self.now)
                age_h = (self.now - m.created_at).total_seconds() / 3600
                rows.append((sim * eff, sim, eff, age_h, m.content))
            return "\n".join(
                f"score={s:.2f}  sim={sim:.2f}  eff={e:.2f}  age={a:.1f}h  │ {c}"
                for s, sim, e, a, c in rows
            )
        return f"Unknown tool: {name}"

    @staticmethod
    def _print_tool(name: str, args: dict, result: str):
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(f"\n  ┌─ {name}({arg_str})")
        for line in result.splitlines():
            print(f"  │  {line}")
        print( "  └─")

    def chat(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            finish = response.choices[0].finish_reason

            if finish == "stop" or not msg.tool_calls:
                return msg.content or ""

            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._dispatch(tc.function.name, args)
                self._print_tool(tc.function.name, args, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })


# ── demo ─────────────────────────────────────────────────────────────────────

def print_store(store: MemoryStore, now: datetime, label: str):
    print(f"\n{'─'*60}")
    print(f"  Memory Snapshot — {label}")
    print(f"{'─'*60}")
    snapshot = store.snapshot(now)
    if not snapshot:
        print("  (empty)")
    for eff, m in snapshot:
        age_days = (now - m.created_at).total_seconds() / 86400
        bar = "█" * int(eff * 20)
        print(f"  {eff:.3f} {bar:<20} [{age_days:.1f}d] {m.content[:60]}")
    print()


def run_demo():
    embedder = Embedder()
    store = MemoryStore(embedder)
    now = datetime.now()

    # ── seed memories at different simulated ages ──────────────────────────
    print("Create memories\n")

    store.add("Alice's birthday is March 15", importance=0.9,
              now=now - timedelta(days=0.1))
    store.add("Team standup is every Monday at 9am", importance=0.7,
              now=now - timedelta(days=2))
    store.add("Bob prefers dark roast coffee", importance=0.4,
              now=now - timedelta(days=3))
    store.add("The project deadline is end of June", importance=1.0,
              now=now - timedelta(days=0.5))
    store.add("Meeting notes: discussed Q3 roadmap", importance=0.3,
              now=now - timedelta(days=7))
    store.add("Parking lot is on level B2", importance=0.2,
              now=now - timedelta(days=14))

    print_store(store, now, "initial state")

    # ── agent conversation ─────────────────────────────────────────────────
    agent = MemoryAgent(store)
    agent.now = now

    conversations = [
        "What do you know about Alice?",
        "When is the project due?",
        "I just learned that the office Wi-Fi password is 'Maple2026!'. Please remember that.",
        "What's the Wi-Fi password?",
    ]

    print("=" * 60)
    print("  Agent Conversation")
    print("=" * 60)
    for user_msg in conversations:
        print(f"\n[USER] {user_msg}")
        reply = agent.chat(user_msg)
        print(f"[AGENT] {reply}")

    print_store(store, now, "final state (after conversation)")


if __name__ == "__main__":
    run_demo()


"""
Create memories
────────────────────────────────────────────────────────────
  Memory Snapshot — initial state
────────────────────────────────────────────────────────────
  0.856 █████████████████    [0.1d] Alice's birthday is March 15
  0.779 ███████████████      [0.5d] The project deadline is end of June
  0.258 █████                [2.0d] Team standup is every Monday at 9am
  0.089 █                    [3.0d] Bob prefers dark roast coffee
  0.009                      [7.0d] Meeting notes: discussed Q3 roadmap
  0.000                      [14.0d] Parking lot is on level B2

============================================================
  Agent Conversation
============================================================

[USER] What do you know about Alice?

  ┌─ recall_memories(query='Alice')
  │  score=0.60  sim=0.67  eff=0.89  age=2.4h  │ Alice's birthday is March 15
  │  score=0.36  sim=0.44  eff=0.81  age=12.0h  │ The project deadline is end of June
  │  score=0.12  sim=0.43  eff=0.27  age=48.0h  │ Team standup is every Monday at 9am
  └─
[AGENT] I know that Alice's birthday is March 15.

[USER] When is the project due?

  ┌─ recall_memories(query='project due date')
  │  score=0.63  sim=0.75  eff=0.84  age=12.0h  │ The project deadline is end of June
  │  score=0.45  sim=0.48  eff=0.93  age=2.4h  │ Alice's birthday is March 15
  │  score=0.17  sim=0.59  eff=0.29  age=48.0h  │ Team standup is every Monday at 9am
  └─
[AGENT] <thought>The `recall_memories` call returned a relevant memory: "The project deadline is end of June". I can now answer the user's question.</thought>The project is due at the end of June.

[USER] I just learned that the office Wi-Fi password is 'Maple2026!'. Please remember that.

  ┌─ store_memory(importance=0.7, content="The office Wi-Fi password is 'Maple2026!'.")
  │  Stored: 'The office Wi-Fi password is 'Maple2026!'.' (importance=0.70, created=2026-06-07 01:03)
  └─
[AGENT] OK. I've remembered that the office Wi-Fi password is 'Maple2026!'.

[USER] What's the Wi-Fi password?

  ┌─ recall_memories(query='Wi-Fi password')
  │  score=0.49  sim=0.66  eff=0.74  age=0.0h  │ The office Wi-Fi password is 'Maple2026!'.
  │  score=0.34  sim=0.35  eff=0.97  age=2.4h  │ Alice's birthday is March 15
  │  score=0.34  sim=0.39  eff=0.87  age=12.0h  │ The project deadline is end of June
  └─
[AGENT] <thought>The `recall_memories` call returned a result stating: "The office Wi-Fi password is 'Maple2026!'." I can now answer the user's question using this information.</thought>The office Wi-Fi password is 'Maple2026!'.

────────────────────────────────────────────────────────────
  Memory Snapshot — final state (after conversation)
────────────────────────────────────────────────────────────
  0.970 ███████████████████  [0.1d] Alice's birthday is March 15
  0.872 █████████████████    [0.5d] The project deadline is end of June
  0.740 ██████████████       [0.0d] The office Wi-Fi password is 'Maple2026!'.
  0.287 █████                [2.0d] Team standup is every Monday at 9am
  0.089 █                    [3.0d] Bob prefers dark roast coffee
  0.009                      [7.0d] Meeting notes: discussed Q3 roadmap
  0.000                      [14.0d] Parking lot is on level B2

"""