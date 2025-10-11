# 🎨 Multimodal RAG System Guide

## Overview

Your RAG system has been upgraded with **multimodal capabilities**, enabling it to understand and process both **text and images** using cutting-edge AI models.

### 🌟 Key Features

1. **GPT-4o Vision Integration** - Analyze and understand images using OpenAI's most advanced vision model
2. **CLIP Embeddings** - Generate semantic embeddings for both text and images in the same vector space
3. **Multimodal Queries** - Ask questions about text documents, images, or combinations of both
4. **Advanced Visualizations** - Interactive charts and plots using Plotly
5. **Beautiful Frontend** - Modern UI with drag-and-drop image upload
6. **API-First Design** - RESTful API with full documentation

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install multimodal requirements
pip install -r requirements.txt

# Note: This will install PyTorch, Transformers, and CLIP models
# The initial download may take some time (~2GB)
```

### 2. Start the Multimodal API Server

```bash
python api_server_multimodal.py
```

The server will start on `http://localhost:8000`

### 3. Create an API Key

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-multimodal-key",
    "user_id": "your-username",
    "rate_limit": 60,
    "expires_in_days": 365
  }'
```

Save the returned API key securely - it's only shown once!

### 4. Open the Frontend

Open `frontend_multimodal.html` in your browser. The frontend includes:

- 📝 Text query interface
- 🖼️ Image upload with drag-and-drop
- 🎨 Multimodal query mode (text + image)
- 📊 Real-time visualizations
- ⚡ Live system status

---

## 🎯 Usage Examples

### Example 1: Text-Only Query

```python
from src.multimodal_rag import MultimodalRAGPipeline
from src.config import RAGConfig

# Initialize pipeline
config = RAGConfig()
pipeline = MultimodalRAGPipeline(config)

# Load existing documents
pipeline.vector_store_manager.load_vector_store()
pipeline._initialize_retriever()

# Query
result = pipeline.query("What is machine learning?")
print(result['answer'])
```

### Example 2: Image Analysis

```python
# Analyze an image using GPT-4 Vision
result = pipeline.analyze_image("path/to/image.jpg")

print(result['analysis'])  # Detailed description
print(result['embedding'][:10])  # CLIP embedding vector (first 10 dims)
print(f"Dimension: {result['embedding_dimension']}")  # 512D
```

### Example 3: Multimodal Query (Text + Image)

```python
# Ask questions about an image with context from documents
result = pipeline.query_multimodal(
    query_text="How does this diagram relate to neural networks?",
    query_image="path/to/diagram.jpg",
    return_source_documents=True
)

print(result['answer'])
print(f"Multimodal: {result['multimodal']}")
```

### Example 4: Using the API

```bash
# Text query
curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is deep learning?"}'

# Image analysis
curl -X POST http://localhost:8000/api/multimodal/analyze-image \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@path/to/image.jpg"

# Multimodal query
curl -X POST http://localhost:8000/api/multimodal/query \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "image=@path/to/image.jpg" \
  -F "query_text=Describe this image"
```

---

## 🏗️ Architecture

### Components

```
┌─────────────────────────────────────────────┐
│         Multimodal RAG System               │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐    ┌──────────────┐     │
│  │   GPT-4o     │    │    CLIP      │     │
│  │   Vision     │    │  Embeddings  │     │
│  └──────────────┘    └──────────────┘     │
│         │                    │              │
│         └────────┬───────────┘              │
│                  │                          │
│        ┌─────────▼─────────┐               │
│        │  Multimodal RAG   │               │
│        │     Pipeline      │               │
│        └─────────┬─────────┘               │
│                  │                          │
│        ┌─────────▼─────────┐               │
│        │   Vector Store    │               │
│        │   (ChromaDB)      │               │
│        └───────────────────┘               │
└─────────────────────────────────────────────┘
```

### Key Classes

#### 1. `MultimodalRAGPipeline`
Main orchestrator for multimodal queries
- Handles text, image, and combined queries
- Integrates with GPT-4o Vision
- Manages embeddings and retrieval

#### 2. `MultimodalEmbeddings`
CLIP-based embedding generator
- Embeds text and images into same vector space
- 512-dimensional embeddings
- Enables semantic similarity across modalities

#### 3. `MultimodalDocument`
Data structure for documents with images
- Text content
- Image data (path or bytes)
- Metadata

---

## 📊 Visualizations

The frontend includes several visualization types:

### 1. **Response Metrics**
Bar charts showing:
- Text length
- Processing time
- Confidence scores

### 2. **Embedding Visualization**
Line plots of embedding vectors:
- First 50 dimensions displayed
- Color-coded by value (positive/negative)
- Interactive hover details

### 3. **System Status**
Real-time metrics:
- Document count
- Query count
- System health
- Current mode

---

## 🎨 Frontend Features

### Three Query Modes

1. **💬 Text Only**
   - Standard RAG queries
   - Uses document retrieval
   - Falls back to general knowledge

2. **🖼️ Image Only**
   - Analyze images with GPT-4 Vision
   - Get CLIP embeddings
   - Detailed visual descriptions

3. **🎨 Text + Image (Multimodal)**
   - Combines text query with image
   - GPT-4 Vision analyzes image in context
   - Retrieves relevant documents
   - Provides comprehensive answers

### Image Upload

- **Drag & Drop** - Drag images directly onto upload zone
- **Click to Browse** - Traditional file picker
- **Preview** - See uploaded image before submitting
- **Supported Formats** - JPG, PNG, WebP

---

## 🔬 Technical Details

### CLIP Model

- **Model**: `openai/clip-vit-base-patch32`
- **Embedding Dimension**: 512
- **Purpose**: Creates unified vector space for text and images
- **Use Case**: Multimodal retrieval and similarity

### GPT-4o Vision

- **Model**: `gpt-4o`
- **Capabilities**:
  - Image understanding
  - Visual question answering
  - Scene description
  - Object detection
  - OCR and text reading

### Vector Storage

- **Backend**: ChromaDB
- **Features**:
  - Persistent storage
  - Fast similarity search
  - Metadata filtering
  - Hybrid search support

---

## 📈 Performance

### Optimization Tips

1. **CLIP Model Loading**
   - First load takes time (~2GB download)
   - Subsequent loads are fast
   - Model cached on disk

2. **GPU Acceleration**
   - CLIP automatically uses GPU if available
   - Falls back to CPU gracefully
   - 10-100x speedup with GPU

3. **Embedding Caching**
   - Embeddings computed once
   - Stored in vector database
   - Instant retrieval after ingestion

4. **Rate Limiting**
   - Configurable per API key
   - Prevents abuse
   - Default: 60 requests/minute

---

## 🧪 Testing

Run the test suite:

```bash
python test_multimodal_rag.py
```

This tests:
1. ✅ Text queries
2. ✅ Image analysis
3. ✅ Multimodal queries
4. ✅ System statistics

---

## 🔌 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Standard text query |
| `/api/multimodal/query` | POST | Multimodal query (text + image) |
| `/api/multimodal/analyze-image` | POST | Analyze single image |
| `/api/stats` | GET | System statistics |
| `/api/visualization/embeddings` | GET | Embedding viz data |

### Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/keys` | POST | Create new API key |
| `/api/keys/info` | GET | Get key information |
| `/api/keys/{id}/stats` | GET | Key usage statistics |

### Documentation

Full interactive API docs: `http://localhost:8000/docs`

---

## 🎯 Use Cases

### 1. Visual Document Analysis
- Upload diagrams, charts, infographics
- Ask questions about visual content
- Extract information from images

### 2. Educational Content
- Explain complex diagrams
- Answer questions about illustrations
- Combine text and visual learning

### 3. Product Catalogs
- Search by image similarity
- Describe products visually
- Match text descriptions to images

### 4. Research & Analysis
- Analyze scientific figures
- Extract data from plots/charts
- Cross-reference text and images

### 5. Accessibility
- Generate image descriptions
- OCR for text in images
- Alternative text generation

---

## 🛠️ Configuration

### Environment Variables

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...your-key...
COHERE_API_KEY=...optional...
```

### RAGConfig Options

```python
from src.config import RAGConfig

config = RAGConfig(
    # LLM Settings
    llm_model="gpt-4o",  # Use GPT-4o for vision
    temperature=0.7,
    max_tokens=1000,
    
    # Embedding Settings
    embedding_model="text-embedding-3-small",
    
    # Retrieval Settings
    top_k_retrieval=5,
    use_hybrid_search=True,
    use_reranking=True,
    
    # Multimodal Settings
    allow_general_knowledge=True,
)
```

---

## 🚨 Troubleshooting

### Issue: CLIP Model Won't Load

**Solution:**
```bash
# Install with specific versions
pip install torch==2.0.0 torchvision==0.15.0
pip install transformers==4.30.0
```

### Issue: Out of Memory with CLIP

**Solution:**
- Use CPU instead of GPU for CLIP
- Process images in smaller batches
- Reduce image resolution before processing

### Issue: GPT-4 Vision API Errors

**Solution:**
- Check OpenAI API key has GPT-4 access
- Verify image size < 20MB
- Ensure image format is supported (JPG, PNG, WebP)

### Issue: Frontend Not Connecting

**Solution:**
- Check API server is running on port 8000
- Disable browser CORS restrictions
- Check API key is valid

---

## 🎓 Best Practices

### 1. Image Quality
- Use high-resolution images for better results
- Clear, well-lit images work best
- Compress large images to < 5MB

### 2. Query Formulation
- Be specific in text queries
- Combine text and image for best results
- Ask one question at a time

### 3. Performance
- Batch process images when possible
- Cache embeddings for repeated queries
- Use GPU for CLIP when available

### 4. Security
- Never commit API keys
- Use environment variables
- Implement rate limiting
- Validate image uploads

---

## 📚 Resources

### Documentation
- [OpenAI GPT-4 Vision](https://platform.openai.com/docs/guides/vision)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)

### Models
- [CLIP on Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32)
- [GPT-4o API](https://platform.openai.com/docs/models/gpt-4o)

---

## 🎉 What's New

### Multimodal Features

✨ **Vision Understanding**
- GPT-4o Vision integration
- Detailed image analysis
- Visual question answering

🎨 **CLIP Embeddings**
- Unified text-image vector space
- Semantic similarity search
- Multimodal retrieval

📊 **Visualizations**
- Interactive Plotly charts
- Embedding visualizations
- Real-time metrics

🖼️ **Enhanced Frontend**
- Drag-and-drop upload
- Three query modes
- Beautiful modern UI

⚡ **Performance**
- GPU acceleration
- Efficient caching
- Parallel processing

---

## 🔮 Future Enhancements

Potential additions:
- [ ] Video analysis support
- [ ] Audio transcription and analysis
- [ ] Multi-image queries
- [ ] Image generation with DALL-E
- [ ] Fine-tuned domain-specific models
- [ ] Advanced visualization dashboards
- [ ] Batch processing API
- [ ] WebSocket for real-time streaming

---

## 💡 Tips & Tricks

1. **Combine Modalities**
   - Use images to clarify ambiguous text
   - Reference images in text queries
   - Cross-validate across modalities

2. **Leverage Context**
   - Provide relevant context in text
   - Use high-quality reference images
   - Ask follow-up questions

3. **Optimize Workflows**
   - Pre-process and cache images
   - Batch similar queries
   - Use appropriate mode for task

4. **Monitor Usage**
   - Track API usage per key
   - Monitor response times
   - Review cache hit rates

---

## 🤝 Contributing

To extend the system:

1. **Add New Vision Models**
   - Implement in `src/multimodal_rag.py`
   - Add to configuration
   - Update API endpoints

2. **Create New Visualizations**
   - Add to `frontend_multimodal.html`
   - Use Plotly for interactivity
   - Follow design patterns

3. **Enhance Embeddings**
   - Try different CLIP variants
   - Experiment with fusion strategies
   - Benchmark performance

---

## 📞 Support

For issues or questions:
- Check troubleshooting section
- Review API documentation
- Test with example scripts
- Verify environment setup

---

## 🎊 Conclusion

Your RAG system is now a **state-of-the-art multimodal AI platform** capable of understanding and processing both text and images. The combination of GPT-4o Vision and CLIP embeddings provides powerful capabilities for:

- 📚 Enhanced document understanding
- 🖼️ Visual analysis and comprehension
- 🎨 Multimodal reasoning
- 📊 Rich visualizations
- ⚡ High-performance queries

**Start exploring the multimodal capabilities today!**

```bash
# Launch the server
python api_server_multimodal.py

# Open the frontend
open frontend_multimodal.html

# Start querying!
```

Happy multimodal AI building! 🚀🎨

