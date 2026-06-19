# Build a RAG Pipeline with Pinecone and OpenAI

**Time to complete:** ~20 minutes  
**What you'll build:** A Python script that embeds a small knowledge base, stores it in Pinecone, retrieves relevant context at query time, and passes it to an LLM to generate a grounded answer.

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a pattern for making LLMs more accurate and up-to-date. Instead of relying solely on what the model learned during training, RAG retrieves relevant information from an external knowledge base at query time and includes it in the prompt.

The pattern has three stages:

```
User question
     │
     ▼
[Embed the question] ──► query vector
     │
     ▼
[Search Pinecone] ──► top-k matching documents
     │
     ▼
[Prompt the LLM] ──► "Answer this question using only the context below: ..."
     │
     ▼
Grounded answer
```

RAG improves accuracy because the LLM answers from documents you control, not from training data that may be outdated or incomplete. It also reduces hallucination by giving the model a factual anchor for its response.

---

## Before you start

You need:

- A Pinecone account and API key ([get one free](https://app.pinecone.io/))
- An OpenAI account and API key ([platform.openai.com](https://platform.openai.com/))
- Python 3.9 or later

---

## Step 1 — Install dependencies

```bash
pip install pinecone openai python-dotenv
```

---

## Step 2 — Set your API keys

```bash
# .env
PINECONE_API_KEY=your_pinecone_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

```bash
echo ".env" >> .gitignore
```

---

## Step 3 — Define your knowledge base

For this walkthrough, the knowledge base is three short documents about a fictional product. In a real application, this would be your documentation, support articles, or any text corpus.

Create `rag_pipeline.py`:

```python
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Knowledge base: documents to embed and store
documents = [
    {
        "id": "doc-001",
        "text": "To cancel your subscription, go to Account Settings, select Billing, and click Cancel Plan. Cancellation takes effect at the end of your current billing period.",
        "metadata": {"topic": "billing", "doc_type": "support"}
    },
    {
        "id": "doc-002",
        "text": "The API rate limit is 1,000 requests per minute on the Pro plan and 100 requests per minute on the Starter plan. Rate limit errors return HTTP 429.",
        "metadata": {"topic": "api", "doc_type": "reference"}
    },
    {
        "id": "doc-003",
        "text": "You can export your data at any time from Account Settings → Data Export. Exports are available in CSV and JSON format and are emailed to your account address within 15 minutes.",
        "metadata": {"topic": "data", "doc_type": "support"}
    },
]
```

---

## Step 4 — Embed the documents

Call the OpenAI Embeddings API to convert each document's text into a vector. The model returns a 1536-dimensional vector for each input.

```python
EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Embed all documents
vectors = []
for doc in documents:
    vector = embed_text(doc["text"])
    vectors.append({
        "id": doc["id"],
        "values": vector,
        "metadata": {**doc["metadata"], "text": doc["text"]}  # store text in metadata for retrieval
    })

print(f"Embedded {len(vectors)} documents")
```

> **Store text in metadata.** Pinecone stores vectors, not text. To retrieve the original text at query time, store it as a metadata field. For long documents, store a truncated excerpt or a reference ID that points to text in a separate store.

---

## Step 5 — Create a Pinecone index and upsert

```python
INDEX_NAME = "rag-demo"
DIMENSION = 1536   # must match text-embedding-3-small output

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
index.upsert(vectors=vectors)
print(f"Upserted {len(vectors)} vectors to index '{INDEX_NAME}'")
```

---

## Step 6 — Retrieve relevant context for a query

At query time, embed the user's question using the same model and query Pinecone for the most similar documents.

```python
def retrieve_context(question: str, top_k: int = 2) -> list[dict]:
    query_vector = embed_text(question)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return [match["metadata"] for match in results["matches"]]
```

The retrieval returns the metadata dictionaries for the closest matches — which includes the original document text.

---

## Step 7 — Generate a grounded answer

Pass the retrieved documents to the LLM as context. The system prompt instructs the model to answer only from the provided context, which reduces hallucination.

```python
def answer_with_rag(question: str) -> str:
    context_docs = retrieve_context(question)
    context_text = "\n\n".join(doc["text"] for doc in context_docs)

    system_prompt = (
        "You are a helpful support assistant. Answer the user's question using only the "
        "context provided below. If the context does not contain enough information to "
        "answer the question, say so — do not make up information.\n\n"
        f"Context:\n{context_text}"
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Run it
question = "How do I cancel my plan?"
answer = answer_with_rag(question)
print(f"\nQ: {question}")
print(f"A: {answer}")
```

Expected output:

```
Q: How do I cancel my plan?
A: To cancel your subscription, go to Account Settings, select Billing, and click Cancel Plan. 
   The cancellation takes effect at the end of your current billing period.
```

---

## What just happened

The pipeline:

1. **Embedded your question** using `text-embedding-3-small` — the same model used to embed the knowledge base, which ensures the vector spaces are comparable
2. **Queried Pinecone** — found the document whose embedding is most similar in direction to the question embedding
3. **Retrieved the text** from that document's metadata
4. **Prompted the LLM** with the retrieved text as context and a constraint to answer only from it
5. **Returned a grounded answer** — the LLM's response is anchored to your documents, not its training data

---

## Common production considerations

**Chunking long documents.** This example uses short documents that fit easily in the context window. For longer content, split documents into overlapping chunks before embedding — typically 256–512 tokens with a 10–20% overlap to avoid splitting mid-concept.

**Metadata filtering.** Add a `filter` to your `index.query()` call to restrict retrieval to a subset of your knowledge base — for example, only returning support articles (`doc_type: "support"`) for user-facing queries.

```python
results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True,
    filter={"doc_type": {"$eq": "support"}}
)
```

**Embedding model consistency.** Always use the same embedding model to embed documents and queries. Mixing models produces incomparable vectors and returns poor results.

**Re-ranking.** For higher-precision applications, add a re-ranking step after retrieval — use a cross-encoder model to re-score the retrieved candidates before passing them to the LLM.

---

## Next steps

- **[What are vector embeddings?](./what-are-vector-embeddings.md)** — understand the concepts behind the embedding and search steps
- **[Zero to First Query — Pinecone Quickstart](./quickstart-pinecone-python.md)** — foundational Pinecone operations without the LLM layer
- **[Pinecone metadata filtering](https://docs.pinecone.io/guides/data/filter-with-metadata)** — filter retrieved results by structured attributes
- **[Pinecone API reference](https://docs.pinecone.io/reference/api/introduction)** — full reference for index and data operations

---

## Troubleshooting

**Retrieved context is irrelevant.**  
Check that you embedded documents and queries with the same model. Also check that `include_metadata=True` is set on the query call and that document text is stored in the metadata dict.

**LLM says "I don't have enough information."**  
Increase `top_k` in the retrieval step to pass more context. Also check that the relevant document is actually in your index — run a direct query and inspect the results before adding the LLM layer.

**`InvalidRequestError` from OpenAI.**  
The context plus the user question may exceed the model's context window. Chunk your documents into smaller segments, or truncate the retrieved text before building the prompt.

**Pinecone index returns empty results.**  
The index may not be ready immediately after creation. Add a brief delay or poll `pc.describe_index(INDEX_NAME).status["ready"]` before upserting.
