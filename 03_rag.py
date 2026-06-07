import os
import re

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 400  # words per chunk


def _fetch_wiki(title: str) -> str:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    return next(iter(pages.values()))["extract"]


def _chunk(text: str, size: int) -> list[str]:
    # split on paragraph breaks first, then merge into ~size-word chunks
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, buf, count = [], [], 0
    for para in paras:
        words = len(para.split())
        if count + words > size and buf:
            chunks.append(" ".join(buf))
            buf, count = [], 0
        buf.append(para)
        count += words
    if buf:
        chunks.append(" ".join(buf))
    return chunks


print("[DATA] Fetching Wikipedia: Large language model...")
_wiki_text = _fetch_wiki("Large_language_model")
DOCUMENTS = _chunk(_wiki_text, CHUNK_SIZE)
print(f"[DATA] {len(DOCUMENTS)} chunks ready.")


def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in sorted(response.data, key=lambda x: (x.index or 0))]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm


def retrieve(query_vec: np.ndarray, doc_vecs: np.ndarray, documents: list[str], k: int = 3) -> list[str]:
    scores = cosine_similarity(query_vec, doc_vecs)
    top_k = np.argsort(scores)[::-1][:k]
    return [(documents[i], float(scores[i])) for i in top_k]


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so. "
    "Be concise and cite which parts of the context support your answer."
)


class RAGAssistant:
    def __init__(self, documents: list[str], k: int = 3):
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_URL"),
        )
        self.model = os.getenv("LLM_MODEL")
        self.k = k
        print(f"[INIT] Model: {self.model}")
        print(f"[INIT] Embedding model: {EMBED_MODEL}")
        print(f"[INIT] Indexing {len(documents)} documents...")
        self.documents = documents
        self.doc_vecs = embed(self.client, documents)
        print(f"[INIT] Index ready. Embedding shape: {self.doc_vecs.shape}")

    def run(self, question: str) -> str:
        print(f"\n{'='*60}")
        print(f"[QUESTION] {question}")
        print(f"{'='*60}")

        query_vec = embed(self.client, [question])[0]
        hits = retrieve(query_vec, self.doc_vecs, self.documents, k=self.k)

        print(f"\n[RETRIEVED {self.k} chunks]")
        context_parts = []
        for i, (chunk, score) in enumerate(hits, 1):
            print(f"  [{i}] score={score:.3f}  {chunk[:80]}...")
            context_parts.append(f"[{i}] {chunk}")
        context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        print("\n[CALLING LLM]")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        answer = response.choices[0].message.content

        print(f"\n{'='*60}")
        print("[ANSWER]")
        print(answer)
        print(f"{'='*60}")
        return answer


if __name__ == "__main__":
    rag = RAGAssistant(DOCUMENTS, k=3)
    rag.run("In which year, prompt injection represents a significant risk?")


"""
[DATA] Fetching Wikipedia: Large language model...
[DATA] 24 chunks ready.
[INIT] Model: gemma-4-31b-it
[INIT] Embedding model: models/gemini-embedding-001
[INIT] Indexing 24 documents...
[INIT] Index ready. Embedding shape: (24, 3072)

============================================================
[QUESTION] In which year, prompt injection represents a significant risk?
============================================================

[RETRIEVED 3 chunks]
  [1] score=0.642  A problem with the primitive dialog or task format is that users can create mess...
  [2] score=0.620  ==== Selection bias ====
Selection bias refers the inherent tendency of large la...
  [3] score=0.619  ==== Content filtering ====
LLM applications accessible to the public, like Chat...

[CALLING LLM]

============================================================
[ANSWER]
<thought>*   Question: "In which year, prompt injection represents a significant risk?"
    *   Constraint: Use ONLY the provided context. Be concise. Cite support.

    *   Context [1]: Discusses prompt injection as a problem where users create messages that appear to come from the assistant/developer, leading to jailbreaking. Mention of "newer models" and "adversarial robustness". No specific year mentioned for "significant risk".
    *   Context [2]: Under the "Safety" section, it states: "As of 2025, prompt injection represents a significant risk to consumers and businesses using agentic features with access to their private data."
    *   Context [3]: Discusses content filtering, "LLM grooming" in 2025, and bypassing GPT-4o's safety controls in 2024.

    *   The specific statement "prompt injection represents a significant risk" is linked to the year 2025 in Context [2].</thought>As of 2025, prompt injection represents a significant risk to businesses and consumers using agentic features that have access to their private data [2].
============================================================
"""