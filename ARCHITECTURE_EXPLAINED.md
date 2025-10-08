# 🏗️ RAG System Architecture - Complete Explanation

## Overview
This document explains how your RAG system integrates OpenAI API, the complete data flow, and how reranking works.

---

## 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY: "What is RAG?"                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      1. CONFIGURATION LAYER                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  config.py - Loads .env file with OpenAI API key           │     │
│  │  ✓ load_dotenv() → Reads OPENAI_API_KEY                    │     │
│  │  ✓ Creates RAGConfig with GPT-4o & OpenAI embeddings       │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    2. RAG PIPELINE INITIALIZATION                    │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  RAGPipeline.__init__()                                     │     │
│  │  ├─ EmbeddingManager (OpenAI API key passed here)          │     │
│  │  │  └─ Creates OpenAIEmbeddings(text-embedding-3-large)    │     │
│  │  ├─ VectorStoreManager                                      │     │
│  │  ├─ AdvancedRetriever                                       │     │
│  │  └─ CohereReranker (optional, if COHERE_API_KEY exists)    │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      3. QUERY PROCESSING FLOW                        │
│                                                                       │
│  Step 3.1: Query Embedding                                           │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  User Query → OpenAI Embeddings API                         │     │
│  │  "What is RAG?" → [0.123, -0.456, 0.789, ... (3072 dims)]  │     │
│  │  Uses: text-embedding-3-large via OpenAI API                │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Step 3.2: Hybrid Search (ENABLED)                                   │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  A) Semantic Search (Vector)                                │     │
│  │     - Compares query embedding with document embeddings     │     │
│  │     - Uses ChromaDB vector store                            │     │
│  │     - Finds semantically similar documents                  │     │
│  │                                                              │     │
│  │  B) Keyword Search (BM25)                                   │     │
│  │     - Traditional keyword matching                          │     │
│  │     - Excels at exact term matches                          │     │
│  │     - Uses rank_bm25 algorithm                              │     │
│  │                                                              │     │
│  │  C) Ensemble Combination                                    │     │
│  │     - Combines both approaches: 50% vector + 50% BM25       │     │
│  │     - Retrieves top_k=5 documents                           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Step 3.3: Contextual Compression (ENABLED)                          │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  LLM-Based Compression (using OpenAI API)                   │     │
│  │  ├─ Takes 5 retrieved documents                             │     │
│  │  ├─ Uses GPT-3.5-turbo to extract relevant parts           │     │
│  │  ├─ Removes irrelevant information                          │     │
│  │  └─ Returns compressed, focused context                     │     │
│  │                                                              │     │
│  │  Result: 5 documents → 5 compressed documents               │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  Step 3.4: Reranking (OPTIONAL - Currently Disabled)                 │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  If COHERE_API_KEY is provided:                             │     │
│  │  ├─ Sends query + 5 documents to Cohere Rerank API         │     │
│  │  ├─ Cohere scores each document's relevance                │     │
│  │  ├─ Returns documents sorted by relevance score            │     │
│  │  └─ Keeps only top rerank_top_k=3 documents                │     │
│  │                                                              │     │
│  │  Without Cohere: Skip this step, use all 5 documents        │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    4. ANSWER GENERATION                              │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  Context: Retrieved & compressed documents                  │     │
│  │  Question: User's original query                            │     │
│  │  ↓                                                           │     │
│  │  Enhanced Prompt Template:                                  │     │
│  │  "You are an intelligent AI assistant..."                   │     │
│  │  "Context: {compressed documents}"                          │     │
│  │  "Question: {user query}"                                   │     │
│  │  "Provide comprehensive, accurate answer..."                │     │
│  │  ↓                                                           │     │
│  │  GPT-4o via OpenAI API                                      │     │
│  │  ├─ Model: gpt-4o                                           │     │
│  │  ├─ Temperature: 0.3 (focused, accurate)                    │     │
│  │  ├─ Max Tokens: 4000 (detailed responses)                   │     │
│  │  └─ API Key: From .env file                                 │     │
│  │  ↓                                                           │     │
│  │  Intelligent, Comprehensive Answer                          │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        5. RESPONSE TO USER                           │
│  {                                                                    │
│    "answer": "Detailed intelligent answer from GPT-4o...",           │
│    "source_documents": [doc1, doc2, doc3, doc4, doc5]                │
│  }                                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 OpenAI API Key Integration

### Where the API Key is Used:

#### 1. **Configuration Layer** (`src/config.py`)
```python
# Lines 10-11: Load .env file
load_dotenv()  # Reads .env file and loads OPENAI_API_KEY

# Line 18: Store in config
openai_api_key: Optional[str] = Field(
    default_factory=lambda: os.getenv("OPENAI_API_KEY")
)
```

**Flow:**
```
.env file → load_dotenv() → os.environ → RAGConfig.openai_api_key
```

#### 2. **Embedding Generation** (`src/embeddings.py`)
```python
# Lines 39-48: Check if OpenAI embeddings requested
if self.model_name.startswith("text-embedding"):
    # OpenAI embeddings require API key
    return OpenAIEmbeddings(
        model=self.model_name,          # "text-embedding-3-large"
        openai_api_key=self.openai_api_key  # From config
    )
```

**API Calls:**
- **When:** During document ingestion & query time
- **What:** Converts text → 3072-dimensional vectors
- **Cost:** ~$0.13 per 1M tokens

#### 3. **Contextual Compression** (`src/retrieval.py`)
```python
# Lines 108-116: LLM-based compression
if self.openai_api_key:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=self.openai_api_key  # From config
    )
    compressor = LLMChainExtractor.from_llm(llm)
```

**API Calls:**
- **When:** For each query, after retrieval
- **What:** Extracts relevant parts from retrieved documents
- **Model:** GPT-3.5-turbo (cheaper for this task)

#### 4. **Answer Generation** (`src/rag_pipeline.py`)
```python
# Lines 190-195: Initialize GPT-4o
llm = ChatOpenAI(
    model_name=self.config.llm_model,      # "gpt-4o"
    temperature=self.config.temperature,    # 0.3
    openai_api_key=self.config.openai_api_key,  # From config
    max_tokens=self.config.max_tokens       # 4000
)
```

**API Calls:**
- **When:** For each query, final step
- **What:** Generates intelligent answer from context
- **Model:** GPT-4o (most intelligent)
- **Cost:** ~$5/1M input tokens, ~$15/1M output tokens

---

## 🎯 Complete Query Flow Example

### Example: User asks "What is RAG?"

```
1. Configuration (happens once at startup)
   └─ .env file contains: OPENAI_API_KEY=sk-proj-...
   └─ load_dotenv() loads it
   └─ RAGConfig stores it

2. Query Arrives
   User: "What is RAG?"
   
3. Embed Query (OpenAI API Call #1)
   Request to: api.openai.com/v1/embeddings
   Model: text-embedding-3-large
   Input: "What is RAG?"
   Output: [0.123, -0.456, ... ] (3072 numbers)
   
4. Hybrid Search
   A) Vector Search:
      - Compare query embedding with 87 document embeddings
      - Find 5 most similar (cosine similarity)
      
   B) BM25 Search:
      - Find documents with keywords: "RAG", "what", "is"
      - Score by frequency & rarity
      
   C) Combine:
      - Merge results with 50/50 weighting
      - Get top 5 documents
      
5. Contextual Compression (OpenAI API Call #2-6)
   For each of 5 documents:
   Request to: api.openai.com/v1/chat/completions
   Model: gpt-3.5-turbo
   Prompt: "Extract relevant parts for: What is RAG?"
   Document: [full document text]
   Output: [compressed, relevant excerpts]
   
6. Reranking (OPTIONAL - Currently Disabled)
   IF Cohere API key exists:
   Request to: api.cohere.ai/v1/rerank
   Query: "What is RAG?"
   Documents: [5 compressed documents]
   Output: Documents sorted by relevance score
   Keep: Top 3 documents
   
7. Generate Answer (OpenAI API Call #7)
   Request to: api.openai.com/v1/chat/completions
   Model: gpt-4o
   Temperature: 0.3
   Max Tokens: 4000
   Prompt:
     "You are an intelligent AI assistant...
      Context: [compressed documents]
      Question: What is RAG?
      Provide comprehensive answer..."
   
   Output: 
     "Retrieval-Augmented Generation (RAG) is an advanced 
      AI technique that enhances large language models by 
      integrating external knowledge retrieval..." 
      [4 detailed paragraphs]
   
8. Return Response
   {
     "answer": "Detailed answer from GPT-4o...",
     "source_documents": [doc1, doc2, doc3, doc4, doc5]
   }
```

---

## 🔄 Reranking Deep Dive

### What is Reranking?

Reranking is a **second-stage refinement** that re-evaluates retrieved documents to improve relevance.

### Why Reranking?

**Problem:** Initial retrieval (vector + BM25) might rank documents by:
- Semantic similarity (vector search)
- Keyword frequency (BM25)

But these don't always capture true relevance to the specific query.

**Solution:** A specialized reranking model evaluates query-document pairs together.

### How Reranking Works in Your System

#### Without Reranking (Current Setup):
```
Query → Hybrid Search → 5 documents → Compression → GPT-4o → Answer
```

#### With Reranking (When Cohere API key is added):
```
Query → Hybrid Search → 5 documents → Compression → 
  Cohere Rerank → Top 3 most relevant → GPT-4o → Answer
```

### Cohere Reranker Implementation

**Location:** `src/retrieval.py`, lines 186-254

```python
class CohereReranker:
    def rerank(self, query: str, documents: List[Document], top_k: int = 3):
        # 1. Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # 2. Call Cohere Rerank API
        results = self.client.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=texts,
            top_n=top_k  # Return only top 3
        )
        
        # 3. Reorder documents by relevance score
        reranked_docs = []
        for result in results.results:
            doc = documents[result.index]
            doc.metadata['rerank_score'] = result.relevance_score
            reranked_docs.append(doc)
        
        return reranked_docs  # Top 3 most relevant
```

### Reranking Process Example

**Input:** 5 documents from hybrid search

| Rank | Document | Initial Score | Content Preview |
|------|----------|---------------|-----------------|
| 1 | Doc A | 0.85 | "RAG combines retrieval..." |
| 2 | Doc B | 0.82 | "Information retrieval systems..." |
| 3 | Doc C | 0.80 | "RAG is a technique..." |
| 4 | Doc D | 0.78 | "Language models use..." |
| 5 | Doc E | 0.75 | "Retrieval augmented..." |

**Cohere Reranking:**
- Analyzes each document in context of the specific query
- Assigns new relevance scores based on semantic meaning
- Considers query-document interaction

**Output:** 3 documents after reranking

| New Rank | Document | Rerank Score | Why Better? |
|----------|----------|--------------|-------------|
| 1 | Doc C | 0.95 | Most direct explanation of RAG |
| 2 | Doc A | 0.91 | Good context on combining retrieval |
| 3 | Doc E | 0.87 | Relevant augmented retrieval info |

Docs B and D are dropped (less relevant to specific query).

### Benefits of Reranking:

1. **Higher Precision:** Focus on most relevant documents
2. **Better Context:** GPT-4o gets better input → better answers
3. **Cost Savings:** Process fewer documents through LLM
4. **Improved Quality:** Cross-encoder models excel at relevance

### How to Enable Reranking:

1. **Get Cohere API key** from https://cohere.ai
2. **Add to .env file:**
   ```bash
   COHERE_API_KEY=your_cohere_key_here
   ```
3. **System automatically enables it** (line 64-68 in rag_pipeline.py)

---

## 📊 Data Flow Summary

### Document Ingestion (One-time)
```
Documents → DocumentProcessor → Chunks →
OpenAI Embeddings API → Vectors →
ChromaDB Vector Store (saved to disk)
```

### Query Processing (Every query)
```
User Query →
┌─ OpenAI Embeddings API (query → vector)
├─ Vector Search (find similar documents)
├─ BM25 Search (keyword matching)
├─ Ensemble (combine results)
├─ OpenAI GPT-3.5 API (compress documents) [5 calls]
├─ [Optional] Cohere Rerank API (improve relevance)
└─ OpenAI GPT-4o API (generate answer) [1 call]
  → Intelligent Answer
```

### API Calls Per Query:
- **1x** OpenAI Embeddings (query embedding)
- **5x** OpenAI GPT-3.5 (contextual compression)
- **0-1x** Cohere Rerank (if enabled)
- **1x** OpenAI GPT-4o (answer generation)

**Total:** ~7 API calls per query (without caching)

---

## 💡 Key Advantages of This Architecture

1. **Hybrid Search:** Catches both semantic and keyword matches
2. **Contextual Compression:** Reduces noise, improves relevance
3. **Optional Reranking:** Further refinement when needed
4. **GPT-4o Generation:** Most intelligent, comprehensive answers
5. **Caching:** Repeated queries served instantly (0 API calls)

---

## 🔧 Current Configuration

```yaml
Embedding Model: text-embedding-3-large (OpenAI)
LLM Model: gpt-4o (OpenAI)
Temperature: 0.3 (focused, accurate)
Max Tokens: 4000 (detailed responses)
Top K Retrieval: 5 documents
Hybrid Search: ✅ ENABLED
Contextual Compression: ✅ ENABLED (LLM-based)
Reranking: ❌ DISABLED (no Cohere key)
Caching: ✅ ENABLED (1 hour TTL)
```

---

## 📈 Performance Characteristics

### Without Reranking (Current):
- **Retrieval:** 5 documents
- **API Calls:** 7 per query
- **Response Time:** ~3-5 seconds
- **Cost:** ~$0.02-0.05 per query
- **Quality:** Very High (GPT-4o + compression)

### With Reranking (If Enabled):
- **Retrieval:** 3 documents (more focused)
- **API Calls:** 8 per query (+ Cohere)
- **Response Time:** ~4-6 seconds
- **Cost:** ~$0.025-0.06 per query
- **Quality:** Excellent (best relevance)

---

## 🎓 Summary

Your RAG system is a **sophisticated, multi-stage pipeline** that:

1. **Loads OpenAI API key** from .env file at startup
2. **Uses OpenAI embeddings** for semantic understanding (3072-dim vectors)
3. **Combines vector + keyword search** for comprehensive retrieval
4. **Compresses context** with GPT-3.5 for relevance
5. **Optionally reranks** with Cohere for precision (when enabled)
6. **Generates answers** with GPT-4o for maximum intelligence

The result: **Highly accurate, comprehensive, intelligent answers** backed by your document knowledge base! 🧠✨

