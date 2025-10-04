# RAG System - Algorithm Details

This document provides detailed explanations of the algorithms used in this RAG system.

## Table of Contents
1. [Document Processing](#document-processing)
2. [Embedding Generation](#embedding-generation)
3. [Vector Storage](#vector-storage)
4. [Retrieval Algorithms](#retrieval-algorithms)
5. [Reranking](#reranking)
6. [Generation](#generation)

## Document Processing

### Chunking Strategies

#### 1. Recursive Character Text Splitting
**Algorithm**: Hierarchical splitting with semantic preservation

```
function recursive_split(text, separators, chunk_size):
    if length(text) <= chunk_size:
        return [text]
    
    for separator in separators:
        if separator in text:
            chunks = split(text, separator)
            result = []
            for chunk in chunks:
                if length(chunk) > chunk_size:
                    result += recursive_split(chunk, separators[1:], chunk_size)
                else:
                    result.append(chunk)
            return result
    
    return [text]
```

**Separators**: `["\n\n", "\n", " ", ""]`

**Advantages**:
- Preserves document structure
- Maintains semantic coherence
- Works well for most document types

**Use Cases**: General-purpose document processing

#### 2. Token-Based Splitting
**Algorithm**: Fixed token count chunking

```
function token_split(text, chunk_size, overlap):
    tokens = tokenize(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(detokenize(chunk_tokens))
    
    return chunks
```

**Advantages**:
- Precise token control
- Consistent chunk sizes
- Better for token-limited models

**Use Cases**: When exact token counts matter

#### 3. Semantic Splitting
**Algorithm**: Embedding-based semantic grouping

```
function semantic_split(text):
    sentences = split_sentences(text)
    embeddings = [embed(s) for s in sentences]
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            embeddings[i],
            mean(embeddings[current_chunk])
        )
        
        if similarity > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(join(current_chunk))
            current_chunk = [sentences[i]]
    
    return chunks
```

**Advantages**:
- Preserves semantic coherence
- Adaptive chunk sizes
- Better topic separation

**Use Cases**: Documents with clear topic transitions

## Embedding Generation

### Vector Representation

Embeddings convert text into high-dimensional vectors where semantic similarity corresponds to geometric proximity.

**Process**:
```
text → tokenization → model forward pass → pooling → normalization → vector
```

### Recommended Models

#### 1. all-mpnet-base-v2
- **Dimensions**: 768
- **Algorithm**: MPNet (Masked and Permuted Pre-training)
- **Training**: Trained on 1B+ sentence pairs
- **Performance**: Best all-around quality/speed balance

#### 2. all-MiniLM-L6-v2
- **Dimensions**: 384
- **Algorithm**: Distilled MiniLM
- **Training**: Knowledge distillation from larger models
- **Performance**: 5x faster, 80% of quality

#### 3. text-embedding-3-small (OpenAI)
- **Dimensions**: 1536
- **Algorithm**: Proprietary transformer
- **Performance**: State-of-the-art quality

### Similarity Calculation

**Cosine Similarity**:
```
similarity(A, B) = (A · B) / (||A|| * ||B||)
```

Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite

## Vector Storage

### ChromaDB
**Algorithm**: SQLite + HNSW (Hierarchical Navigable Small World)

**HNSW Overview**:
- Builds a multi-layer graph of vectors
- Layer 0: All vectors
- Upper layers: Sparse connections for fast navigation
- Query time: O(log N)

**Advantages**:
- Easy persistence
- Built-in metadata filtering
- Good for development

### FAISS (Facebook AI Similarity Search)
**Algorithm**: Multiple index types

**IndexFlatL2**: Exact search using L2 distance
- Time: O(N)
- Space: O(N*d)
- Use: Small datasets, exact results needed

**IndexIVFFlat**: Inverted file with flat quantization
- Partitions space into Voronoi cells
- Time: O(N/k) where k = num partitions
- Use: Medium datasets

**IndexHNSW**: Hierarchical Navigable Small World
- Best performance for large-scale
- Approximate but very accurate
- Use: Production systems

## Retrieval Algorithms

### 1. Semantic Search (Vector Similarity)

**Algorithm**:
```
function semantic_search(query, k):
    query_embedding = embed(query)
    
    # Find k nearest neighbors
    similarities = []
    for doc_embedding in vector_store:
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc, sim))
    
    # Return top k
    return sort(similarities, reverse=True)[:k]
```

**Complexity**: O(N) for exact, O(log N) for HNSW

**Strengths**:
- Finds semantically similar content
- Works with paraphrasing
- Language understanding

**Weaknesses**:
- May miss exact keyword matches
- Requires good embeddings

### 2. BM25 (Keyword Search)

**Algorithm**: Best Matching 25 (probabilistic ranking)

```
BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / 
             (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

where:
- D: Document
- Q: Query
- qi: Query term i
- f(qi, D): Frequency of qi in D
- |D|: Length of document D
- avgdl: Average document length
- k1: Term frequency saturation (typical: 1.2-2.0)
- b: Length normalization (typical: 0.75)
- IDF(qi): Inverse document frequency
```

**IDF Calculation**:
```
IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))

where:
- N: Total documents
- n(qi): Documents containing qi
```

**Strengths**:
- Excellent for exact matches
- Fast
- Interpretable scores

**Weaknesses**:
- No semantic understanding
- Requires exact keyword matches

### 3. Hybrid Search

**Algorithm**: Ensemble retrieval with weighted combination

```
function hybrid_search(query, k, alpha=0.5):
    # Semantic results
    semantic_docs = semantic_search(query, k*2)
    semantic_scores = normalize_scores(semantic_docs)
    
    # Keyword results
    keyword_docs = bm25_search(query, k*2)
    keyword_scores = normalize_scores(keyword_docs)
    
    # Combine scores
    combined = {}
    for doc, score in semantic_scores:
        combined[doc] = alpha * score
    
    for doc, score in keyword_scores:
        combined[doc] = combined.get(doc, 0) + (1 - alpha) * score
    
    # Return top k
    return sort(combined.items(), reverse=True)[:k]
```

**Alpha parameter**:
- 0.0: Pure keyword search
- 0.5: Equal weight (default)
- 1.0: Pure semantic search

**Advantages**:
- Best of both worlds
- Robust to different query types
- Higher recall

### 4. Contextual Compression

**Algorithm**: Filter irrelevant content from retrieved documents

**LLM-Based Compression**:
```
function llm_compress(query, documents):
    compressed = []
    
    for doc in documents:
        prompt = f"""
        Extract only the parts relevant to: {query}
        
        Document: {doc.content}
        
        Relevant excerpts:
        """
        
        relevant_parts = llm.generate(prompt)
        compressed.append(relevant_parts)
    
    return compressed
```

**Embedding-Based Compression**:
```
function embedding_compress(query, documents, threshold=0.76):
    query_embedding = embed(query)
    filtered = []
    
    for doc in documents:
        # Split into sentences
        sentences = split_sentences(doc)
        
        # Keep relevant sentences
        relevant = []
        for sentence in sentences:
            sent_embedding = embed(sentence)
            similarity = cosine_similarity(query_embedding, sent_embedding)
            
            if similarity > threshold:
                relevant.append(sentence)
        
        if relevant:
            filtered.append(join(relevant))
    
    return filtered
```

**Advantages**:
- Removes noise
- Reduces token usage
- Improves answer quality

## Reranking

### Cohere Rerank API

**Algorithm**: Cross-encoder for query-document relevance

**Cross-Encoder vs Bi-Encoder**:

**Bi-Encoder** (used for initial retrieval):
```
embed(query) → vector_q
embed(doc) → vector_d
score = similarity(vector_q, vector_d)
```

**Cross-Encoder** (used for reranking):
```
concatenate(query, doc) → joint_input
score = model(joint_input)  # Direct relevance prediction
```

**Process**:
```
function rerank(query, documents, k):
    # Send to Cohere API
    response = cohere.rerank(
        model="rerank-english-v2.0",
        query=query,
        documents=[doc.content for doc in documents],
        top_n=k
    )
    
    # Reorder by relevance scores
    reranked = []
    for result in response.results:
        doc = documents[result.index]
        doc.metadata['rerank_score'] = result.relevance_score
        reranked.append(doc)
    
    return reranked
```

**Advantages**:
- Higher accuracy than vector similarity
- Better understanding of relevance
- Improves final results quality

**Tradeoff**:
- Slower than vector search
- Must be run on smaller candidate set
- Requires API calls

**Recommended Pipeline**:
1. Retrieve 10-20 candidates (fast vector search)
2. Rerank to top 3-5 (accurate but slower)

## Generation

### Retrieval-Augmented Generation Process

```
function rag_generate(query, documents, llm):
    # Create context from retrieved documents
    context = ""
    for i, doc in enumerate(documents):
        context += f"\n\n[Document {i+1}]\n{doc.content}"
    
    # Build prompt
    prompt = f"""
    Use the following documents to answer the question.
    If the answer is not in the documents, say so.
    
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate response
    answer = llm.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return answer
```

### Prompt Engineering

**Key Components**:

1. **Task Definition**: Clear instruction
2. **Context**: Retrieved documents
3. **Query**: User's question
4. **Constraints**: Guidelines (e.g., "don't make up information")
5. **Format**: Expected response structure

**Example Template**:
```
You are a helpful assistant that answers questions using provided context.

Instructions:
- Use ONLY information from the context below
- If the answer is not in the context, say "I don't have enough information"
- Cite sources when possible
- Be concise but complete

Context:
{retrieved_documents}

Question: {user_question}

Answer:
```

## Performance Optimization

### Chunking
- **Small chunks (500-800)**: Better precision, more specific retrieval
- **Large chunks (1000-1500)**: Better context, fewer chunks to process

### Retrieval
- **Top K**: Start with 5-10, adjust based on results
- **Hybrid search**: +20-30% recall vs semantic alone
- **Reranking**: +10-15% accuracy

### Embeddings
- **Model size**: Larger = better quality, slower
- **Normalization**: Always normalize for cosine similarity
- **Caching**: Cache embeddings to avoid recomputation

### Vector Store
- **FAISS**: 5-10x faster than ChromaDB for large datasets
- **HNSW**: Best balance of speed and accuracy
- **Batch operations**: Process in batches for efficiency

## References

1. **RAG**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. **BM25**: Robertson & Zaragoza (2009) - The Probabilistic Relevance Framework: BM25 and Beyond
3. **HNSW**: Malkov & Yashunin (2016) - Efficient and robust approximate nearest neighbor search
4. **Sentence Transformers**: Reimers & Gurevych (2019) - Sentence-BERT
5. **FAISS**: Johnson et al. (2019) - Billion-scale similarity search with GPUs

