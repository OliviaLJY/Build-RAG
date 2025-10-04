# RAG System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼───────┐      ┌───────▼────────┐
            │   INGESTION   │      │    QUERY       │
            │    PHASE      │      │    PHASE       │
            └───────┬───────┘      └───────┬────────┘
                    │                      │
        ┌───────────┴───────────┐         │
        │                       │         │
        ▼                       ▼         ▼
┌───────────────┐      ┌──────────────┐  ┌──────────────┐
│   DOCUMENT    │      │  EMBEDDING   │  │  RETRIEVAL   │
│  PROCESSING   │─────▶│  GENERATION  │  │    ENGINE    │
└───────────────┘      └──────┬───────┘  └──────┬───────┘
                              │                  │
                              ▼                  ▼
                       ┌─────────────┐    ┌─────────────┐
                       │   VECTOR    │    │  RERANKING  │
                       │    STORE    │    │  (Optional) │
                       └─────────────┘    └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │     LLM     │
                                          │  GENERATION │
                                          └─────────────┘
```

## Component Architecture

### 1. Document Processing Layer

```
┌─────────────────────────────────────────────────┐
│           DOCUMENT PROCESSOR                     │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐      ┌──────────────┐        │
│  │   LOADERS    │      │  CHUNKING    │        │
│  │              │      │  STRATEGIES  │        │
│  │  • PDF       │─────▶│              │        │
│  │  • TXT       │      │  • Recursive │        │
│  │  • DOCX      │      │  • Token     │        │
│  │  • Directory │      │  • Semantic  │        │
│  └──────────────┘      └──────┬───────┘        │
│                               │                 │
│                               ▼                 │
│                        ┌──────────────┐        │
│                        │   METADATA   │        │
│                        │   ENRICHMENT │        │
│                        └──────────────┘        │
└─────────────────────────────────────────────────┘
```

**Algorithms**:
- Recursive Character Splitting: Hierarchical semantic preservation
- Token-based Splitting: Fixed token count chunks
- Semantic Splitting: Embedding-based grouping

### 2. Embedding Layer

```
┌─────────────────────────────────────────────────┐
│           EMBEDDING MANAGER                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────┐    ┌──────────────────┐  │
│  │  HUGGINGFACE     │    │     OPENAI       │  │
│  │  EMBEDDINGS      │    │   EMBEDDINGS     │  │
│  │                  │    │                  │  │
│  │  • all-mpnet-v2  │    │  • text-embed-3  │  │
│  │  • MiniLM-L6     │    │  • ada-002       │  │
│  │  • Multilingual  │    │                  │  │
│  └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │             │
│           └───────────┬───────────┘             │
│                       │                         │
│                       ▼                         │
│              ┌─────────────────┐               │
│              │  NORMALIZATION  │               │
│              │  & POOLING      │               │
│              └─────────────────┘               │
└─────────────────────────────────────────────────┘
```

**Models**:
- all-mpnet-base-v2: 768 dims, best quality/speed
- all-MiniLM-L6-v2: 384 dims, fastest
- text-embedding-3-small: 1536 dims, highest quality

### 3. Vector Store Layer

```
┌─────────────────────────────────────────────────┐
│          VECTOR STORE MANAGER                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐           ┌──────────────┐   │
│  │   CHROMADB   │           │    FAISS     │   │
│  │              │           │              │   │
│  │  • SQLite    │           │  • IndexFlat │   │
│  │  • HNSW      │           │  • IndexIVF  │   │
│  │  • Metadata  │           │  • IndexHNSW │   │
│  │  • Filter    │           │              │   │
│  └──────┬───────┘           └──────┬───────┘   │
│         │                          │            │
│         └──────────┬───────────────┘            │
│                    │                            │
│                    ▼                            │
│           ┌─────────────────┐                  │
│           │  PERSISTENCE    │                  │
│           │  & INDEXING     │                  │
│           └─────────────────┘                  │
└─────────────────────────────────────────────────┘
```

**Algorithms**:
- HNSW: Hierarchical Navigable Small World graphs
- IVF: Inverted File Index with clustering
- Cosine Similarity: Vector proximity measurement

### 4. Retrieval Engine

```
┌─────────────────────────────────────────────────────────┐
│              ADVANCED RETRIEVER                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐         ┌─────────────────┐       │
│  │   SEMANTIC      │         │    KEYWORD      │       │
│  │   SEARCH        │         │    SEARCH       │       │
│  │                 │         │                 │       │
│  │  • Vector       │         │  • BM25         │       │
│  │  • Cosine Sim   │         │  • TF-IDF       │       │
│  │  • Top K        │         │  • Exact Match  │       │
│  └────────┬────────┘         └────────┬────────┘       │
│           │                           │                 │
│           └───────────┬───────────────┘                 │
│                       │                                 │
│                       ▼                                 │
│              ┌─────────────────┐                       │
│              │ HYBRID ENSEMBLE │                       │
│              │  (α semantic +  │                       │
│              │   β keyword)    │                       │
│              └────────┬────────┘                       │
│                       │                                 │
│                       ▼                                 │
│              ┌─────────────────┐                       │
│              │  CONTEXTUAL     │                       │
│              │  COMPRESSION    │                       │
│              │  • LLM-based    │                       │
│              │  • Embed-based  │                       │
│              └────────┬────────┘                       │
│                       │                                 │
│                       ▼                                 │
│              ┌─────────────────┐                       │
│              │   RERANKING     │                       │
│              │  • Cohere API   │                       │
│              │  • Cross-encode │                       │
│              └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

**Pipeline Flow**:
1. Query → Embeddings
2. Vector Search (semantic) + BM25 (keyword)
3. Hybrid combination (weighted)
4. Contextual compression (filter irrelevant)
5. Reranking (improve ordering)
6. Top K results

### 5. RAG Pipeline (Orchestration)

```
┌──────────────────────────────────────────────────────┐
│                  RAG PIPELINE                         │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐                                    │
│  │    INPUT     │                                    │
│  │   Question   │                                    │
│  └──────┬───────┘                                    │
│         │                                            │
│         ▼                                            │
│  ┌──────────────┐      ┌──────────────┐            │
│  │   RETRIEVE   │─────▶│   RERANK     │            │
│  │  Documents   │      │  (Optional)  │            │
│  └──────────────┘      └──────┬───────┘            │
│                               │                     │
│                               ▼                     │
│                        ┌──────────────┐            │
│                        │   CONTEXT    │            │
│                        │  FORMATION   │            │
│                        └──────┬───────┘            │
│                               │                     │
│                               ▼                     │
│                        ┌──────────────┐            │
│                        │    PROMPT    │            │
│                        │  ENGINEERING │            │
│                        └──────┬───────┘            │
│                               │                     │
│                               ▼                     │
│                        ┌──────────────┐            │
│                        │     LLM      │            │
│                        │  GENERATION  │            │
│                        └──────┬───────┘            │
│                               │                     │
│                               ▼                     │
│                        ┌──────────────┐            │
│                        │    OUTPUT    │            │
│                        │   Answer +   │            │
│                        │   Sources    │            │
│                        └──────────────┘            │
└──────────────────────────────────────────────────────┘
```

## Data Flow

### Ingestion Flow

```
Documents
   │
   ├─▶ Load (PDF/TXT/DOCX)
   │
   ├─▶ Extract Text
   │
   ├─▶ Chunk (Recursive/Token/Semantic)
   │      ├─ Size: 500-1500 tokens
   │      └─ Overlap: 10-20%
   │
   ├─▶ Generate Embeddings
   │      └─ Sentence Transformers / OpenAI
   │
   ├─▶ Store in Vector DB
   │      ├─ ChromaDB (HNSW index)
   │      └─ FAISS (Multiple index types)
   │
   └─▶ Add Metadata
          ├─ Source
          ├─ Chunk ID
          └─ Custom fields
```

### Query Flow

```
User Query
   │
   ├─▶ Embed Query
   │
   ├─▶ Retrieve Candidates
   │      ├─ Semantic Search (top 10-20)
   │      │     └─ Cosine similarity
   │      │
   │      └─ Keyword Search (top 10-20)
   │            └─ BM25 algorithm
   │
   ├─▶ Hybrid Ensemble
   │      └─ Weighted combination (α=0.5)
   │
   ├─▶ Contextual Compression (optional)
   │      ├─ LLM-based filtering
   │      └─ Embedding similarity filter
   │
   ├─▶ Reranking (optional)
   │      └─ Cohere cross-encoder (top 3-5)
   │
   ├─▶ Construct Context
   │      └─ Format retrieved documents
   │
   ├─▶ Generate Prompt
   │      ├─ System instructions
   │      ├─ Context documents
   │      └─ User question
   │
   ├─▶ LLM Generation
   │      └─ GPT-3.5/4 or other
   │
   └─▶ Return Answer + Sources
```

## Configuration Hierarchy

```
RAGConfig
   │
   ├─▶ Document Processing
   │      ├─ chunk_size: 1000
   │      ├─ chunk_overlap: 200
   │      └─ chunking_strategy: "recursive"
   │
   ├─▶ Embeddings
   │      └─ embedding_model: "all-mpnet-base-v2"
   │
   ├─▶ Vector Store
   │      ├─ vector_store_type: "chromadb"
   │      ├─ collection_name: "rag_documents"
   │      └─ persist_directory: "./data/vectorstore"
   │
   ├─▶ Retrieval
   │      ├─ top_k_retrieval: 5
   │      ├─ rerank_top_k: 3
   │      ├─ use_hybrid_search: True
   │      ├─ use_reranking: True
   │      └─ use_contextual_compression: True
   │
   └─▶ Generation
          ├─ llm_model: "gpt-3.5-turbo"
          ├─ temperature: 0.7
          └─ max_tokens: 2000
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Document Loading | O(N) | N = file size |
| Chunking | O(N) | Linear scan |
| Embedding | O(N*D) | N = chunks, D = model depth |
| Vector Store Insert | O(log N) | HNSW index |
| Semantic Search | O(log N) | HNSW approximate |
| BM25 Search | O(N) | Can be optimized with index |
| Reranking | O(K) | K = candidates (small) |
| LLM Generation | O(T) | T = output tokens |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Raw Documents | O(N) | Original size |
| Chunks | O(N) | ~same as original |
| Embeddings | O(C*D) | C = chunks, D = dimensions |
| Vector Index | O(C*D) | Plus index overhead |
| BM25 Index | O(V) | V = vocabulary size |

## Scalability Considerations

### Small Scale (< 1K documents)
- Use: ChromaDB with default settings
- Embedding: any model
- Retrieval: Simple semantic search sufficient

### Medium Scale (1K - 100K documents)
- Use: ChromaDB or FAISS with HNSW
- Embedding: Consider smaller models for speed
- Retrieval: Hybrid search recommended

### Large Scale (100K+ documents)
- Use: FAISS with IVF or HNSW
- Embedding: Batch processing, caching
- Retrieval: Hybrid + aggressive filtering
- Consider: Distributed vector stores

## Module Dependencies

```
rag_pipeline.py
    ├── config.py
    ├── document_processor.py
    ├── embeddings.py
    ├── vector_store.py
    └── retrieval.py
        ├── vector_store.py
        └── embeddings.py

embeddings.py
    └── [langchain, sentence-transformers, openai]

vector_store.py
    └── [chromadb, faiss, langchain]

document_processor.py
    └── [langchain, pypdf, python-docx, unstructured]

retrieval.py
    └── [langchain, cohere]
```

## Extension Points

### 1. Custom Document Loaders
Add new loaders in `document_processor.py`:
```python
def _load_custom_format(self, file_path: str):
    # Your custom loading logic
    return documents
```

### 2. Custom Chunking Strategies
Implement in `document_processor.py`:
```python
class CustomTextSplitter:
    def split_text(self, text):
        # Your chunking logic
        return chunks
```

### 3. Custom Retrievers
Extend in `retrieval.py`:
```python
class CustomRetriever:
    def retrieve(self, query):
        # Your retrieval logic
        return documents
```

### 4. Custom Rerankers
Add reranking methods:
```python
class CustomReranker:
    def rerank(self, query, documents):
        # Your reranking logic
        return reranked_documents
```

## Security Considerations

1. **API Keys**: Store in environment variables, never commit
2. **Document Access**: Implement access control if needed
3. **Input Validation**: Sanitize user queries
4. **Output Filtering**: Filter sensitive information
5. **Rate Limiting**: Implement for production APIs

## Monitoring & Observability

Key metrics to track:
- Retrieval latency
- Embedding generation time
- LLM response time
- Relevance scores
- User feedback
- Cache hit rates
- Vector store size

## Future Enhancements

Potential improvements:
1. Multi-modal support (images, tables)
2. Streaming responses
3. Conversation memory
4. Query optimization
5. Automated evaluation
6. A/B testing framework
7. Distributed processing
8. Real-time document updates

