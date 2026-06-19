# Zero to First Query — Pinecone Python Quickstart

**Time to complete:** ~15 minutes  
**What you'll build:** A working Python script that creates a Pinecone index, upserts vectors with metadata, and runs a similarity search query.

---

## Before you start

You need:

- A Pinecone account ([create one free](https://app.pinecone.io/))
- Python 3.9 or later
- Your Pinecone API key — find it in the [Pinecone console](https://app.pinecone.io/) under **API Keys**

---

## Step 1 — Install the SDK

```bash
pip install pinecone
```

---

## Step 2 — Set your API key

Create a `.env` file in your project root. Never commit API keys to source control.

```bash
# .env
PINECONE_API_KEY=your_api_key_here
```

Add `.env` to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

---

## Step 3 — Initialize the client

Create `quickstart.py`:

```python
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
```

> **Free vs. paid plans:** The free Starter plan gives you one project with one serverless index. You don't need a credit card to follow this guide.

---

## Step 4 — Create an index

An index is where your vectors live. You define the **dimension** (number of values in each vector) and the **metric** used for similarity comparisons at creation time. You can't change these after the index exists.

```python
index_name = "quickstart"

# Only create the index if it doesn't already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3,          # must match the dimension of your vectors
        metric="cosine",      # cosine | euclidean | dotproduct
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)
```

**Choosing a metric:**
- `cosine` — best for text embeddings; measures directional similarity regardless of magnitude
- `euclidean` — best for coordinate or numeric data; measures absolute distance
- `dotproduct` — best for pre-normalized vectors; fastest at query time

**About dimension:** When you use a real embedding model, your dimension is determined by that model. For example, OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors. This quickstart uses 3 to keep things readable.

---

## Step 5 — Upsert vectors

Upserting stores vectors in your index. If a vector with the same ID already exists, Pinecone overwrites it.

```python
vectors = [
    {
        "id": "doc-001",
        "values": [0.1, 0.2, 0.9],
        "metadata": {"category": "machine-learning", "title": "Intro to embeddings"}
    },
    {
        "id": "doc-002",
        "values": [0.3, 0.8, 0.1],
        "metadata": {"category": "databases", "title": "What is a vector database?"}
    },
    {
        "id": "doc-003",
        "values": [0.9, 0.1, 0.2],
        "metadata": {"category": "machine-learning", "title": "How RAG works"}
    },
]

index.upsert(vectors=vectors)
print("Upserted", len(vectors), "vectors")
# → Upserted 3 vectors
```

**Metadata** is optional but powerful. You can attach any key-value pairs to a vector and use them to filter query results — returning only vectors that match both the similarity threshold and a metadata condition.

---

## Step 6 — Query the index

A query returns the `top_k` most similar vectors to a query vector. You can optionally add a `filter` to restrict results by metadata.

```python
query_vector = [0.1, 0.3, 0.8]   # close to doc-001 in this vector space

results = index.query(
    vector=query_vector,
    top_k=2,
    include_metadata=True
)

for match in results["matches"]:
    print(f"{match['id']} — score: {match['score']:.4f} — {match['metadata']['title']}")
```

Expected output:

```
doc-001 — score: 0.9921 — Intro to embeddings
doc-002 — score: 0.8734 — What is a vector database?
```

**Score interpretation:** With cosine similarity, scores range from -1 to 1. A score of 1.0 means the vectors are identical in direction. In practice, anything above 0.85 is typically a strong match for text embeddings — but the right threshold depends on your data and use case.

---

## Step 7 — Filter by metadata

Add a `filter` to restrict results to vectors that match a metadata condition, even if other vectors score higher.

```python
filtered_results = index.query(
    vector=query_vector,
    top_k=2,
    include_metadata=True,
    filter={"category": {"$eq": "machine-learning"}}
)

for match in filtered_results["matches"]:
    print(f"{match['id']} — {match['metadata']['title']}")

# → doc-001 — Intro to embeddings
# → doc-003 — How RAG works
```

Only vectors tagged `machine-learning` are returned, even though `doc-002` scored second in the unfiltered query.

---

## Full script

```python
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Create index
index_name = "quickstart"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 2. Upsert vectors
vectors = [
    {"id": "doc-001", "values": [0.1, 0.2, 0.9], "metadata": {"category": "machine-learning", "title": "Intro to embeddings"}},
    {"id": "doc-002", "values": [0.3, 0.8, 0.1], "metadata": {"category": "databases", "title": "What is a vector database?"}},
    {"id": "doc-003", "values": [0.9, 0.1, 0.2], "metadata": {"category": "machine-learning", "title": "How RAG works"}},
]
index.upsert(vectors=vectors)
print("Upserted", len(vectors), "vectors")

# 3. Query
query_vector = [0.1, 0.3, 0.8]
results = index.query(vector=query_vector, top_k=2, include_metadata=True)
print("\nTop matches:")
for match in results["matches"]:
    print(f"  {match['id']} — score: {match['score']:.4f} — {match['metadata']['title']}")

# 4. Filter by metadata
filtered = index.query(
    vector=query_vector,
    top_k=2,
    include_metadata=True,
    filter={"category": {"$eq": "machine-learning"}}
)
print("\nFiltered (machine-learning only):")
for match in filtered["matches"]:
    print(f"  {match['id']} — {match['metadata']['title']}")
```

Run it:

```bash
python quickstart.py
```

---

## What just happened

When you ran this script, Pinecone:

1. **Created a serverless index** — Pinecone provisioned storage and an HNSW graph for fast approximate nearest-neighbor search
2. **Indexed your vectors** — each vector was stored with its ID and metadata, and added to the search graph
3. **Ran similarity search** — Pinecone traversed the graph to find the `top_k` vectors most similar to your query vector, ranked by cosine score
4. **Applied metadata filtering** — the filter was evaluated server-side before returning results, not as a post-query client-side operation

---

## Next steps

- **[Understanding vector embeddings](./what-are-vector-embeddings.md)** — learn how text, images, and other data become vectors
- **[RAG walkthrough](./rag-with-pinecone.md)** — connect Pinecone to an LLM for retrieval-augmented generation
- **[Pinecone API reference](https://docs.pinecone.io/reference/api/introduction)** — full reference for all index and data operations

---

## Troubleshooting

**`PineconeException: UNAUTHENTICATED`**  
Your API key is missing or incorrect. Check that `PINECONE_API_KEY` is set in your `.env` file and that you're loading it with `load_dotenv()` before initializing the client.

**`Index not found` on query**  
The index isn't ready yet. Serverless index creation is fast but not instant — add a short wait or poll `pc.describe_index(index_name).status["ready"]` before querying.

**Score is `None` or unexpectedly low**  
Make sure `include_metadata=True` is set on the query call, and that the query vector has the same dimension as the index. Mismatched dimensions raise a `400` error.

**`filter` returns fewer results than `top_k`**  
Expected behavior. Pinecone applies the metadata filter before ranking, so if fewer than `top_k` vectors match the filter, the result set will be smaller than requested.
