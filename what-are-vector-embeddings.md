# What Are Vector Embeddings?

**Read time:** ~10 minutes  
**Who this is for:** Developers familiar with APIs who are new to machine learning or vector search.

---

## The problem embeddings solve

Traditional databases are great at exact matches. You can ask "find all records where `category = 'sports'"` and get a precise result. But they can't answer questions like "find me content *similar in meaning* to this sentence" — because they have no concept of meaning, only structure.

Embeddings solve this by converting data — text, images, audio, or anything else — into lists of numbers called **vectors**. The key property of a well-trained embedding model is that data with similar meaning produces vectors that are numerically close to each other.

This lets you search by semantic similarity instead of keyword matching. A query for "how do I cancel my account" can return a support article titled "Deleting your profile" even though it shares no words with the query.

---

## What a vector actually looks like

An embedding is just a list of floating-point numbers. For example, the sentence "machine learning is useful" might become:

```
[0.023, -0.418, 0.891, 0.204, -0.071, ... ]  # 1536 values total
```

Each number represents a dimension in the model's learned vector space. The dimensions don't have intuitive labels — they emerge from training on large datasets. What matters is that the relationships between vectors reflect relationships in meaning.

**Dimensions by model:**

| Model | Dimensions |
|---|---|
| OpenAI `text-embedding-3-small` | 1536 |
| OpenAI `text-embedding-3-large` | 3072 |
| Cohere `embed-english-v3.0` | 1024 |
| Google `textembedding-gecko` | 768 |

A vector database like Pinecone stores these vectors and lets you search them at scale.

---

## How similarity is measured

Two vectors are "similar" if they point in roughly the same direction in vector space. There are three common ways to measure this:

### Cosine similarity

Measures the angle between two vectors, ignoring their magnitude. A score of 1.0 means the vectors are identical in direction. A score of 0 means they're unrelated. A score of -1 means they're opposite.

**Best for:** Text embeddings. Cosine is the standard choice for semantic search because it's not affected by the length of the original text (which would change vector magnitude but not direction).

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

### Euclidean distance

Measures the straight-line distance between two points in vector space. Smaller values mean more similar.

**Best for:** Numeric or coordinate data where magnitude matters, such as geospatial search or image feature vectors.

### Dot product

Computes the product of two vectors' magnitudes and the cosine of the angle between them. Fastest at query time, but only reliable when vectors are pre-normalized to the same magnitude.

**Best for:** High-performance scenarios where you control the embedding pipeline and normalize your vectors before upsert.

---

## How embedding models work (simplified)

An embedding model is a neural network trained to place similar inputs near each other in vector space and dissimilar inputs far apart.

For text models, this training typically uses large datasets of text where the model learns to:

- Place synonyms and paraphrases close together
- Place antonyms far apart
- Cluster content by topic
- Preserve semantic relationships (e.g., "king" minus "man" plus "woman" ≈ "queen")

You don't need to train your own embedding model. Pretrained models from OpenAI, Cohere, Google, and Hugging Face convert your input to vectors through a single API call.

**Example with OpenAI:**

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    input="machine learning is useful",
    model="text-embedding-3-small"
)

vector = response.data[0].embedding
print(len(vector))   # → 1536
print(vector[:5])    # → [0.023, -0.418, 0.891, 0.204, -0.071]
```

---

## Where vector databases come in

Storing and searching vectors at scale is a different problem than traditional database queries. A vector database like Pinecone is purpose-built for it.

The core challenge: finding the most similar vectors to a query by doing brute-force comparisons against every stored vector is prohibitively slow at scale — comparing a query against 10 million vectors one by one takes too long for real-time applications.

Pinecone uses **approximate nearest neighbor (ANN)** algorithms — specifically a data structure called **HNSW (Hierarchical Navigable Small World)** — to find similar vectors in milliseconds without comparing against every stored vector. The tradeoff is that results are approximate, not exact. In practice, the accuracy is high enough for nearly all production use cases.

```
Query vector ──► HNSW graph ──► top-k approximate nearest neighbors
                               (ranked by cosine, euclidean, or dot product)
```

---

## Sparse vs. dense vectors

The vectors described above are **dense** — every dimension has a value. Dense vectors are what embedding models produce and are used for semantic similarity search.

**Sparse vectors** have mostly zero values and represent keyword presence, like a traditional TF-IDF representation. A sparse vector for "machine learning" might be:

```
{4821: 0.73, 12004: 0.61, ...}   # only non-zero dimensions stored
```

Sparse vectors are better at exact keyword matching. Dense vectors are better at semantic matching. **Hybrid search** combines both — returning results that score well on meaning *and* keyword relevance — and is increasingly the standard approach for production search systems.

---

## When to use vector search vs. traditional search

| | Vector search | Traditional (keyword) search |
|---|---|---|
| Query: "how do I cancel my account" → "Deleting your profile" | ✅ | ❌ |
| Query: exact phrase match | ✅ (if trained) | ✅ |
| Query: product SKU lookup | ❌ | ✅ |
| Semantic recommendations | ✅ | ❌ |
| Filtering by date range, category | ✅ with metadata | ✅ |
| Speed at 10M+ records | ✅ with ANN | ✅ |

Most production systems use both. Vector search handles semantic and recommendation queries; traditional indexes handle exact lookups and structured filters.

---

## Next steps

- **[Zero to First Query — Pinecone Python Quickstart](./quickstart-pinecone-python.md)** — create an index, upsert vectors, and run your first similarity search
- **[RAG with Pinecone](./rag-with-pinecone.md)** — use vector search to give an LLM accurate, up-to-date context
- **[Pinecone metadata filtering](https://docs.pinecone.io/guides/data/filter-with-metadata)** — combine vector similarity with structured filters for precise results
