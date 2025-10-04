# RAG System - Retrieval-Augmented Generation

A production-ready Retrieval-Augmented Generation (RAG) system with state-of-the-art algorithms for document ingestion, semantic search, and intelligent question answering.

## 🎯 Features

### Core Capabilities
- **Multiple Document Formats**: PDF, TXT, DOCX support
- **Advanced Chunking Strategies**: Recursive, token-based, and semantic chunking
- **Flexible Vector Stores**: ChromaDB and FAISS support
- **Hybrid Search**: Combines semantic search with BM25 keyword matching
- **Contextual Compression**: Filters irrelevant content from retrieved documents
- **Reranking**: Optional Cohere reranking for improved relevance
- **Multiple Embedding Models**: HuggingFace and OpenAI embeddings

### Key Algorithms

1. **Document Processing**
   - Recursive Character Text Splitting (best for most cases)
   - Token-aware splitting for precise control
   - Semantic chunking using sentence transformers

2. **Retrieval**
   - **Semantic Search**: Vector similarity using cosine distance
   - **BM25**: Keyword-based retrieval (Best Matching 25 algorithm)
   - **Hybrid Search**: Ensemble retriever combining both approaches
   - **Contextual Compression**: LLM-based or embeddings-based filtering

3. **Embeddings**
   - `all-mpnet-base-v2`: Best all-around performance (default)
   - `all-MiniLM-L6-v2`: Fast and efficient
   - OpenAI `text-embedding-3-small`: High quality commercial option

4. **Reranking**
   - Cohere's rerank API for post-retrieval scoring
   - Significantly improves relevance of final results

## 📦 Installation

1. **Clone or download this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model** (optional, for advanced text processing):
```bash
python -m spacy download en_core_web_sm
```

4. **Set up API keys** (if using OpenAI or Cohere):

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
```

## 🚀 Quick Start

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Initialize configuration
config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    top_k_retrieval=5
)

# Create RAG pipeline
rag = RAGPipeline(config)

# Ingest documents
rag.ingest_documents("./documents", create_new=True)

# Query
result = rag.query("What is the main topic?")
print(result['answer'])
```

### Run Examples

```bash
# Basic usage
python examples/basic_usage.py

# Advanced features
python examples/advanced_usage.py

# Interactive chat
python examples/interactive_chat.py
```

## 📚 Project Structure

```
RAG/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── document_processor.py  # Document loading and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # Vector store management
│   ├── retrieval.py           # Advanced retrieval algorithms
│   └── rag_pipeline.py        # Main RAG pipeline
├── examples/
│   ├── basic_usage.py         # Basic usage example
│   ├── advanced_usage.py      # Advanced features demo
│   └── interactive_chat.py    # Interactive chat interface
├── documents/                 # Place your documents here
├── data/                      # Vector store persistence
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Key configuration options in `src/config.py`:

```python
RAGConfig(
    # Chunking
    chunk_size=1000,              # Size of text chunks
    chunk_overlap=200,            # Overlap between chunks
    
    # Embeddings
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    
    # LLM
    llm_model="gpt-3.5-turbo",
    temperature=0.7,
    
    # Retrieval
    top_k_retrieval=5,            # Documents to retrieve
    rerank_top_k=3,               # Documents after reranking
    use_hybrid_search=True,       # Enable hybrid search
    use_reranking=True,           # Enable reranking
    use_contextual_compression=True,
    
    # Vector Store
    vector_store_type="chromadb",  # or "faiss"
    persist_directory="./data/vectorstore"
)
```

## 📖 Usage Guide

### 1. Document Ingestion

```python
# From directory
rag.ingest_documents("./documents", create_new=True)

# Add more documents later
rag.ingest_documents("./more_docs", create_new=False)

# From specific file
rag.ingest_documents("./document.pdf", create_new=True)
```

### 2. Querying

```python
# Simple query
answer = rag.chat("What is RAG?")

# Query with sources
result = rag.query("Explain the key concepts", return_source_documents=True)
print(result['answer'])
for doc in result['source_documents']:
    print(doc.page_content)
```

### 3. Custom Document Processing

```python
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=100,
    chunking_strategy="semantic"  # or "recursive", "token"
)

documents = processor.process_text("Your text here")
```

### 4. Direct Similarity Search

```python
# Without LLM generation
documents = rag.retrieve("search query")
for doc in documents:
    print(doc.page_content)
```

### 5. Metadata Filtering

```python
# Add documents with metadata
from langchain.docstore.document import Document

docs = [
    Document(page_content="Python tutorial", metadata={"category": "programming"}),
    Document(page_content="ML basics", metadata={"category": "ai"})
]

# Search with filter
results = rag.vector_store_manager.similarity_search(
    "tutorial",
    k=5,
    filter_dict={"category": "programming"}
)
```

## 🎓 Algorithm Details

### Chunking Strategies

1. **Recursive Character Splitting** (Recommended)
   - Splits on paragraph boundaries first, then sentences
   - Preserves semantic coherence
   - Best for most document types

2. **Token-based Splitting**
   - Splits based on token count
   - Ensures consistent chunk sizes for token-limited models
   - Better for precise token management

3. **Semantic Splitting**
   - Uses sentence transformers to group semantically similar content
   - Most advanced but slower

### Retrieval Pipeline

```
Query → Embedding → Vector Search ──┐
                                    ├→ Hybrid Search → Compression → Reranking → Top K
Query → Tokenization → BM25 ────────┘
```

1. **Vector Search**: Finds semantically similar documents using cosine similarity
2. **BM25**: Finds documents with keyword matches
3. **Hybrid**: Combines both with weighted scores (default: 0.5/0.5)
4. **Compression**: Filters irrelevant content
5. **Reranking**: Re-scores using Cohere API for better relevance

### Embedding Models Comparison

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-mpnet-base-v2 | 768 | Medium | High | General purpose (default) |
| all-MiniLM-L6-v2 | 384 | Fast | Good | Large datasets, speed priority |
| text-embedding-3-small | 1536 | Fast | High | Commercial, high quality |

## 🔬 Advanced Features

### Custom Retriever

```python
from src.retrieval import AdvancedRetriever

retriever = AdvancedRetriever(
    vector_retriever=base_retriever,
    documents=documents,
    use_hybrid_search=True,
    use_compression=True,
    top_k=5
)

results = retriever.retrieve("query")
```

### Reranking

```python
from src.retrieval import CohereReranker

reranker = CohereReranker(api_key="your_key")
reranked = reranker.rerank(query, documents, top_k=3)
```

## 📊 Performance Tips

1. **Chunk Size**: 
   - Smaller (500-800): Better precision, more chunks
   - Larger (1000-1500): Better context, fewer chunks

2. **Overlap**:
   - 10-20% of chunk size recommended
   - Prevents information loss at boundaries

3. **Top K**:
   - Start with 5-10 for retrieval
   - Rerank to 3-5 for final results

4. **Vector Store**:
   - ChromaDB: Better for development, easy persistence
   - FAISS: Faster for large-scale production

5. **Embeddings**:
   - Use local models (HuggingFace) for privacy/cost
   - Use OpenAI for best quality with API costs

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `chunk_size`
   - Use `all-MiniLM-L6-v2` instead of larger models
   - Process documents in batches

2. **Slow Performance**
   - Switch to FAISS for vector store
   - Use smaller embedding model
   - Disable compression/reranking

3. **Poor Results**
   - Increase `top_k_retrieval`
   - Enable hybrid search
   - Add reranking
   - Adjust chunk size/overlap

## 📄 License

MIT License - feel free to use this project for any purpose.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional document loaders
- More embedding models
- Alternative reranking methods
- Performance optimizations
- Additional examples

## 📞 Support

For issues or questions:
1. Check the examples in `/examples`
2. Review configuration in `src/config.py`
3. Read the troubleshooting section above

## 🔗 References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

