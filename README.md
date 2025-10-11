# 🎨 Multimodal RAG System - Vision + Text Intelligence

> **Next-Generation AI**: A production-ready Retrieval-Augmented Generation (RAG) system with **multimodal capabilities** - combining GPT-4o Vision, CLIP embeddings, and advanced text understanding.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

<img width="2064" height="1560" alt="image" src="https://github.com/user-attachments/assets/47b821ba-9259-4094-897a-5b8d57d9f41c" />

---

## 🌟 **NEW: Multimodal Capabilities**

Your RAG system now understands **both text and images**!

<img width="3278" height="1560" alt="image" src="https://github.com/user-attachments/assets/29522df2-b8e1-490b-8123-30f04e22ce81" />

<img width="3288" height="1550" alt="image" src="https://github.com/user-attachments/assets/ee57bdb3-ad95-48bc-a791-b76c7ac0bb8e" />




### ✨ What's New

- 🤖 **GPT-4o Vision Integration** - Analyze and understand images with state-of-the-art AI
- 🎨 **CLIP Embeddings** - Unified 512D vector space for text and images
- 💬 **Three Query Modes** - Text only, Image only, or Combined (Multimodal)
- 📊 **Interactive Visualizations** - Beautiful Plotly charts and embedding plots
- ⚡ **Production API** - FastAPI with authentication, rate limiting, and CORS
- 🖼️ **Modern Frontend** - Drag-and-drop image upload with live results

### 🚀 Quick Multimodal Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Launch multimodal server
./launch_multimodal.sh
```

Then open `frontend_multimodal.html` in your browser!

**📚 Full Guide:** See [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md) for complete documentation.

---

## 🎯 Features

### Core Capabilities

#### Text Understanding (Enhanced)
- **Multiple Document Formats**: PDF, TXT, DOCX support
- **Advanced Chunking Strategies**: Recursive, token-based, and semantic chunking
- **Flexible Vector Stores**: ChromaDB and FAISS support
- **Hybrid Search**: Combines semantic search with BM25 keyword matching
- **Contextual Compression**: Filters irrelevant content from retrieved documents
- **Reranking**: Optional Cohere reranking for improved relevance
- **Multiple Embedding Models**: HuggingFace and OpenAI embeddings

#### Vision & Multimodal (NEW! 🆕)
- **GPT-4o Vision Analysis**: Understand and describe images with AI
- **CLIP Embeddings**: Unified text-image vector space (512D)
- **Image Upload**: Drag-and-drop interface with preview
- **Multimodal Queries**: Combine text questions with images
- **Visual Q&A**: Ask questions about uploaded images
- **OCR Capabilities**: Extract text from images
- **Chart Understanding**: Analyze graphs, diagrams, and visualizations

#### Production Features
- **REST API**: Full FastAPI server with Swagger docs
- **Authentication**: API key management with rate limiting
- **Frontend**: Modern web UI with visualizations
- **Caching**: Query and embedding caching for performance
- **Monitoring**: Usage tracking and statistics

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

### Option 1: Multimodal RAG (Recommended) 🆕

```python
from src.multimodal_rag import MultimodalRAGPipeline
from src.config import RAGConfig

# Initialize multimodal pipeline
pipeline = MultimodalRAGPipeline(RAGConfig())

# Load existing documents
pipeline.load_existing_store()

# 1. Text query
result = pipeline.query("What is machine learning?")
print(result['answer'])

# 2. Image analysis
result = pipeline.analyze_image("diagram.jpg")
print(result['analysis'])

# 3. Multimodal query (text + image)
result = pipeline.query_multimodal(
    query_text="Explain this neural network diagram",
    query_image="nn_diagram.jpg"
)
print(result['answer'])
```

### Option 2: Text-Only RAG (Classic)

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

### Multimodal API Server

```bash
# Start the multimodal API server
python api_server_multimodal.py

# Or use the launcher
./launch_multimodal.sh
```

Then:
1. Open `frontend_multimodal.html` in browser
2. Create API key: `python create_test_key.py`
3. Enter API key in frontend
4. Start querying with text and/or images!

**API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

### Run Examples

```bash
# Text-only examples
python examples/basic_usage.py
python examples/advanced_usage.py
python examples/interactive_chat.py

# Multimodal examples
python demo_multimodal.py          # Interactive demo
python test_multimodal_rag.py      # Test suite
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
│   ├── rag_pipeline.py        # Text-only RAG pipeline
│   └── multimodal_rag.py      # 🆕 Multimodal RAG pipeline (text + vision)
│
├── examples/
│   ├── basic_usage.py         # Basic usage example
│   ├── advanced_usage.py      # Advanced features demo
│   └── interactive_chat.py    # Interactive chat interface
│
├── Multimodal Components 🆕
│   ├── api_server_multimodal.py    # FastAPI server with multimodal endpoints
│   ├── frontend_multimodal.html    # Modern web UI with image upload
│   ├── demo_multimodal.py          # Interactive multimodal demo
│   ├── test_multimodal_rag.py      # Multimodal test suite
│   ├── create_test_key.py          # API key generator
│   └── launch_multimodal.sh        # One-command launcher
│
├── Documentation 📚
│   ├── README.md                   # This file
│   ├── MULTIMODAL_GUIDE.md         # Complete multimodal documentation
│   ├── QUICK_START_MULTIMODAL.md   # 5-minute quick start
│   ├── IMPROVEMENTS_SUMMARY.md     # What's new summary
│   └── FEATURES_OVERVIEW.md        # Detailed feature breakdown
│
├── documents/                 # Place your documents here
├── data/                      # Vector store persistence
│   ├── vectorstore/          # Text embeddings
│   └── api_keys.db           # API key database
├── requirements.txt
└── practice_auth.py          # Authentication system
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

## 🎨 Multimodal Features Guide

### Three Query Modes

| Mode | Input | Best For | Example |
|------|-------|----------|---------|
| 💬 **Text Only** | Text question | Document Q&A | "What is machine learning?" |
| 🖼️ **Image Only** | Image file | Image analysis | Analyze a chart/diagram |
| 🎨 **Multimodal** | Text + Image | Visual Q&A | "Explain this diagram" + image |

### Using the Multimodal API

```bash
# Text query
curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?"}'

# Image analysis
curl -X POST http://localhost:8000/api/multimodal/analyze-image \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@photo.jpg"

# Multimodal query
curl -X POST http://localhost:8000/api/multimodal/query \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@diagram.jpg" \
  -F "query_text=Explain this diagram"
```

### API Key Management

```bash
# Create a new API key
python create_test_key.py

# Or via API
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{"name":"my-key","user_id":"user123"}'
```

### Frontend Features

Open `frontend_multimodal.html` to access:
- 🖼️ **Drag & Drop Upload** - Easy image upload
- 🎯 **Mode Selector** - Switch between text/image/multimodal
- 📊 **Live Visualizations** - Plotly charts and embedding plots
- ⚡ **Real-time Status** - System health monitoring
- 🎨 **Beautiful UI** - Modern gradient design

### Use Cases

1. **📚 Education** - Students upload homework diagrams for explanations
2. **🔬 Research** - Analyze scientific figures and charts
3. **🏥 Healthcare** - Medical image analysis with context
4. **📊 Business** - Understand data visualizations and reports
5. **🎓 Documentation** - Interactive technical docs with images

### Architecture

```
User Query (Text/Image)
    ↓
┌─────────────────────────────────┐
│  MultimodalRAGPipeline          │
│  ┌──────────────────────────┐   │
│  │ GPT-4o Vision            │   │  ← Image understanding
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ CLIP Embeddings (512D)   │   │  ← Text + Image vectors
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Document Retrieval       │   │  ← Context from docs
│  └──────────────────────────┘   │
└─────────────────────────────────┘
    ↓
Comprehensive Answer
```

### Performance

| Operation | Time | Model |
|-----------|------|-------|
| Text Query | 1-2s | GPT-4o + Embeddings |
| Image Analysis | 2-3s | GPT-4o Vision + CLIP |
| Multimodal Query | 3-4s | Full pipeline |

**GPU Acceleration**: CLIP runs 10-100x faster with GPU

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

### Multimodal Issues 🆕

4. **CLIP Model Won't Load**
   ```bash
   pip install torch==2.0.0 torchvision==0.15.0
   pip install transformers==4.30.0
   ```

5. **GPT-4 Vision Errors**
   - Verify API key has GPT-4 access
   - Check image size < 20MB
   - Use supported formats (JPG, PNG, WebP)

6. **API Key Authentication Failed**
   - Create new key: `python create_test_key.py`
   - Verify key starts with `rag_`
   - Check key hasn't expired

## 📄 License

MIT License - feel free to use this project for any purpose.

## 🤝 Contributing

Contributions welcome! Areas for improvement:

**Text RAG:**
- Additional document loaders
- More embedding models
- Alternative reranking methods
- Performance optimizations

**Multimodal (NEW!):**
- Additional vision models (LLaVA, BLIP-2)
- Video analysis support
- Audio transcription integration
- Multi-image queries
- Advanced visualization dashboards

**General:**
- Additional examples and tutorials
- Performance benchmarks
- Integration guides

## 📞 Support

### Documentation
- **Quick Start**: [QUICK_START_MULTIMODAL.md](QUICK_START_MULTIMODAL.md)
- **Complete Guide**: [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md)
- **What's New**: [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)
- **Features**: [FEATURES_OVERVIEW.md](FEATURES_OVERVIEW.md)

### Examples
- Text examples: `/examples` directory
- Multimodal demo: `python demo_multimodal.py`
- Test suite: `python test_multimodal_rag.py`

### API Documentation
- Start server: `python api_server_multimodal.py`
- Visit: `http://localhost:8000/docs`

### For Issues
1. Check the documentation above
2. Review configuration in `src/config.py`
3. Read the troubleshooting section
4. Run test scripts to verify setup

## 🔗 References

### Core Technologies
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

### Multimodal AI
- [OpenAI GPT-4 Vision](https://platform.openai.com/docs/guides/vision)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP](https://openai.com/blog/clip/)

### Research Papers
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Retrieval-Augmented Generation](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)

### Frameworks
- [FastAPI](https://fastapi.tiangolo.com/)
- [Plotly](https://plotly.com/python/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)

---

## 🎉 Get Started Now!

```bash
# Quick launch (3 commands)
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-your-key" > .env
./launch_multimodal.sh
```

Then open `frontend_multimodal.html` and experience the power of multimodal AI! 🚀🎨

