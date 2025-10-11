# 🚀 Quick Start - Multimodal RAG

Get your multimodal RAG system up and running in 5 minutes!

## Step 1: Install Dependencies (2 min)

```bash
# Install all required packages
pip install -r requirements.txt

# This installs:
# - PyTorch & Transformers (for CLIP)
# - FastAPI & Uvicorn (API server)
# - Plotly (visualizations)
# - All RAG components
```

**Note**: First installation may take time (~2GB for CLIP model)

## Step 2: Set Up API Key (1 min)

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=sk-your-openai-key-here" > .env
```

Or set environment variable:

```bash
export OPENAI_API_KEY=sk-your-openai-key-here
```

## Step 3: Launch Server (1 min)

### Option A: Using Launch Script (Recommended)

```bash
./launch_multimodal.sh
```

### Option B: Direct Launch

```bash
python api_server_multimodal.py
```

The server starts on: `http://localhost:8000`

## Step 4: Create API Key (30 sec)

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-test-key",
    "user_id": "demo-user"
  }'
```

**Save the returned API key!** (Example: `rag_a1b2c3d4...`)

## Step 5: Open Frontend (30 sec)

Simply open `frontend_multimodal.html` in your browser:

```bash
# On Mac
open frontend_multimodal.html

# On Linux
xdg-open frontend_multimodal.html

# On Windows
start frontend_multimodal.html
```

Or drag the file into your browser.

## ✨ You're Ready!

Now you can:

1. **Enter your API key** in the frontend
2. **Choose query mode**:
   - 💬 Text Only - Ask about documents
   - 🖼️ Image Only - Analyze images
   - 🎨 Multimodal - Combine text + image

3. **Upload an image** (drag & drop or click)
4. **Ask questions** and see AI-powered answers!

---

## 📸 Quick Test

### Test 1: Image Analysis

1. Upload any image (photo, diagram, chart)
2. Click "Analyze Image Only"
3. See GPT-4 Vision's detailed description
4. View CLIP embedding visualization

### Test 2: Text Query

1. Type: "What is machine learning?"
2. Click "Submit Query"
3. Get answer with sources (if documents loaded)

### Test 3: Multimodal Query

1. Upload an image of a diagram
2. Type: "Explain this diagram in detail"
3. Click "Submit Query"
4. Get comprehensive multimodal answer!

---

## 🎯 What Can You Do?

### Text Capabilities
- ✅ Query your document library
- ✅ Get AI-powered answers with sources
- ✅ Semantic search across documents
- ✅ General knowledge fallback

### Vision Capabilities
- ✅ Analyze images with GPT-4 Vision
- ✅ Describe visual content
- ✅ Extract text from images (OCR)
- ✅ Identify objects and scenes

### Multimodal Capabilities
- ✅ Ask questions about images
- ✅ Combine text and visual context
- ✅ Cross-reference documents and images
- ✅ Generate comprehensive answers

### Visualizations
- ✅ Embedding vector plots
- ✅ Response metrics charts
- ✅ Real-time system status
- ✅ Interactive Plotly graphs

---

## 🔧 Troubleshooting

### Problem: Dependencies won't install

**Solution:**
```bash
# Use Python 3.9+ 
python --version  # Check version

# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: CLIP model download fails

**Solution:**
```bash
# Download manually
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Problem: Port 8000 already in use

**Solution:**
```bash
# Kill existing process
lsof -ti:8000 | xargs kill -9

# Or use different port
python api_server_multimodal.py --port 8001
```

### Problem: API key not working

**Solution:**
- Verify key is correct (starts with `rag_`)
- Create new key if needed
- Check server logs for errors

### Problem: Frontend not connecting

**Solution:**
- Ensure server is running (`http://localhost:8000`)
- Check browser console for errors
- Try different browser (Chrome recommended)
- Disable ad blockers

---

## 📚 Next Steps

### 1. Ingest Your Documents

```python
from src.multimodal_rag import MultimodalRAGPipeline
from src.config import RAGConfig

pipeline = MultimodalRAGPipeline(RAGConfig())
pipeline.ingest_documents("./your-documents-folder/")
```

### 2. Try Advanced Features

- Upload different image types (diagrams, photos, screenshots)
- Combine text queries with images
- Explore visualization features
- Test with complex questions

### 3. API Integration

```python
import requests

url = "http://localhost:8000/api/multimodal/query"
headers = {"X-API-Key": "your-api-key"}
files = {"image": open("image.jpg", "rb")}
data = {"query_text": "What's in this image?"}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json()["answer"])
```

### 4. Explore Documentation

- Read `MULTIMODAL_GUIDE.md` for detailed docs
- Check `/docs` endpoint for API reference
- Review example code in `test_multimodal_rag.py`

---

## 💡 Tips for Best Results

### Image Quality
- Use clear, high-resolution images
- Good lighting and contrast
- Relevant to your query

### Query Formulation
- Be specific and clear
- Ask one thing at a time
- Provide context when needed

### Performance
- First query loads models (slower)
- Subsequent queries are fast
- GPU accelerates CLIP significantly

---

## 🎉 Examples to Try

### Example 1: Analyze a Chart
Upload a chart/graph and ask:
- "What trends do you see in this data?"
- "Explain the key insights from this chart"

### Example 2: Diagram Understanding
Upload a technical diagram and ask:
- "Explain how this system works"
- "What are the main components?"

### Example 3: Photo Analysis
Upload any photo and ask:
- "Describe this scene in detail"
- "What objects can you identify?"

### Example 4: Document + Image
Upload a diagram and ask:
- "How does this relate to machine learning concepts?"
- "Compare this to information in the knowledge base"

---

## ✅ Checklist

Before you start:

- [ ] Python 3.9+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] OpenAI API key configured
- [ ] Server running (`python api_server_multimodal.py`)
- [ ] API key created (via `/api/keys` endpoint)
- [ ] Frontend opened (`frontend_multimodal.html`)
- [ ] API key entered in frontend

---

## 🚀 Launch Commands Summary

```bash
# One-time setup
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-xxx" > .env

# Every time you start
./launch_multimodal.sh

# Create API key
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{"name":"test","user_id":"me"}'

# Open frontend
open frontend_multimodal.html
```

---

## 🎊 You're All Set!

Your multimodal RAG system is ready to:
- 📝 Answer questions from your documents
- 🖼️ Analyze and understand images
- 🎨 Combine text and vision for powerful insights
- 📊 Visualize results beautifully

**Happy querying!** 🚀

---

## 📞 Need Help?

1. Check `MULTIMODAL_GUIDE.md` for detailed documentation
2. Review API docs at `http://localhost:8000/docs`
3. Run tests: `python test_multimodal_rag.py`
4. Check server logs for errors

**System Architecture:**

```
┌─────────────────────────────────────────┐
│  Frontend (HTML + JavaScript)           │
│  - Image upload                         │
│  - Text queries                         │
│  - Visualizations                       │
└──────────────┬──────────────────────────┘
               │ HTTP/REST API
┌──────────────▼──────────────────────────┐
│  API Server (FastAPI)                   │
│  - Authentication                       │
│  - Rate limiting                        │
│  - Request handling                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Multimodal RAG Pipeline                │
│  ┌────────────────────────────────────┐ │
│  │ GPT-4o Vision  │  CLIP Embeddings │ │
│  └────────────────┴──────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │   Document Retrieval System        │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

Everything is integrated and ready to use! 🎉

