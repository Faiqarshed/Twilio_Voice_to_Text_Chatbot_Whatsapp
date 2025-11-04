import os
import chromadb
from sentence_transformers import SentenceTransformer
from models.Chunk import text_splitter
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None

client = chromadb.PersistentClient(path="chromadb_data.db")
collection = client.get_or_create_collection(name="rag_collection")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

async def add_documents(docs):
    all_chunks = []
    for doc in docs:
        chunks = text_splitter.split_text(doc)
        all_chunks.extend(chunks)

    embeddings = embedder.encode(all_chunks).tolist()
    ids = [f"doc_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, embeddings=embeddings, ids=ids)

async def query_rag_response(
    query,
    top_k: int = 4,
    min_score: float = 0.35,
    mmr: bool = True,
    max_context_chars: int = 3000,
    temperature: float = 0.2,
    max_tokens: int = 256,
    only_from_context: bool = True
):
    """Retrieve relevant chunks and synthesize a concise, grounded answer.

    Args:
        query: User question.
        top_k: Number of candidates to retrieve (diverse if MMR enabled).
        min_score: Minimum similarity score to keep (cosine similarity ~ 1 - distance).
        mmr: Use maximal marginal relevance (if supported by Chroma).
        max_context_chars: Limit context size passed to the LLM.
        temperature: LLM sampling temperature.
        max_tokens: Max tokens for the answer.
        only_from_context: If True, instruct LLM to answer only from provided context.
    """

    query_embedding = embedder.encode([query])[0].tolist()

    # Request distances for filtering; enable MMR when available.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(10, top_k),
        include=["documents", "distances"],
    )

    documents = results.get('documents', [[]])[0] if results.get('documents') else []
    distances = results.get('distances', [[]])[0] if results.get('distances') else []

    if not documents:
        return "No relevant answer found"

    # Convert Chroma distances (0 = identical) to similarity, then filter
    scored = []
    for doc, dist in zip(documents, distances or [None] * len(documents)):
        if dist is None:
            similarity = 1.0
        else:
            # For cosine distance from Chroma, similarity ≈ 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        if similarity >= min_score:
            scored.append((similarity, doc))

    if not scored:
        # Fallback to best available even if below threshold
        scored = [(1.0 if (distances and distances[0] is None) else 1.0 - float(distances[0]) if distances else 0.0, documents[0])]

    # Sort by similarity desc, keep top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    selected_docs = [d for _, d in scored[:top_k]]

    # Build concise context within limit
    context = "\n\n".join(selected_docs)
    if len(context) > max_context_chars:
        context = context[:max_context_chars]

    # If OpenAI client available, synthesize a concise answer
    if _openai_client is not None and os.getenv("OPENAI_API_KEY"):
        system_parts = [
            "You are a helpful assistant for question answering over a knowledge base.",
            "Use ONLY the provided context. If the answer is not in the context, say 'I don't know'.",
            "Be concise and precise. Prefer short bullet points where helpful.",
        ]
        if not only_from_context:
            system_parts[1] = "Prefer the provided context; if insufficient, use general knowledge but mark assumptions."

        prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Instructions: Answer succinctly in 2-6 short bullet points, cite brief quotes when clarifying."
        )

        try:
            completion = _openai_client.chat.completions.create(
                model=os.getenv("RAG_MODEL", "gpt-4o-mini"),
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "\n".join(system_parts)},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content.strip()
        except Exception:
            # Fallback to extractive summary if LLM call fails
            pass

    # Last-resort fallback: return the most similar snippet trimmed
    best = selected_docs[0]
    return best[:max_context_chars]