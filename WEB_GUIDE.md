## 🌐 Web Frontend Guide

## Quick Start

### Method 1: Using Launch Script (Easiest)

```bash
./launch_web.sh
```

### Method 2: Direct Command

```bash
streamlit run app.py
```

The web app will automatically open in your browser at `http://localhost:8501`

---

## ✨ Features

### 1. **Interactive Chat Interface**
- Ask questions about AI, Machine Learning, and Deep Learning
- Get instant answers from the knowledge base
- View source passages for transparency

### 2. **Document Upload**
- Upload your own PDF or TXT files
- System automatically processes and indexes them
- Ask questions about your uploaded documents

### 3. **Sample Questions**
Pre-loaded questions in the sidebar:
- "What is machine learning?"
- "Explain neural networks"
- "What is deep learning?"
- "How does backpropagation work?"
- And more!

### 4. **Source Attribution**
- Every answer shows the source passages
- Click "View Sources" to see where the information came from
- Multiple relevant passages displayed

### 5. **System Information**
- Real-time stats on loaded documents
- Current model information
- System status

---

## 📚 Pre-loaded Knowledge Base

The system comes with comprehensive AI learning materials:

1. **RAG Introduction** (`rag_introduction.txt`)
   - What is RAG
   - How it works
   - Key algorithms
   - Best practices

2. **Machine Learning Basics** (`machine_learning_basics.txt`)
   - Types of ML (supervised, unsupervised, reinforcement)
   - Common algorithms
   - Model evaluation
   - Best practices

3. **Deep Learning Guide** (`deep_learning_guide.txt`)
   - Neural network fundamentals
   - CNNs, RNNs, Transformers
   - Training techniques
   - Optimization algorithms

4. **Neural Networks Explained** (`neural_networks_explained.txt`)
   - Network architectures
   - Activation functions
   - Backpropagation
   - Regularization

---

## 🎯 How to Use

### Ask Questions

Type your question in the chat input at the bottom:

**Good questions:**
- "What is the difference between supervised and unsupervised learning?"
- "Explain how convolutional neural networks work"
- "What are the benefits of using LSTM over RNN?"
- "How does the attention mechanism work?"
- "What is gradient descent?"

**The system will:**
1. Search through all documents
2. Find the most relevant passages
3. Display the best answer
4. Show additional related sources

### Upload Your Documents

1. Click the "Upload Documents" section in the sidebar
2. Click "Browse files" and select PDF or TXT files
3. Click "Process Uploaded Documents"
4. Wait for processing to complete
5. Start asking questions about your documents!

### Use Sample Questions

Click any sample question in the sidebar to instantly ask it. Great for exploring the knowledge base!

### Clear Chat History

Click the "🗑️ Clear Chat" button in the sidebar to start a fresh conversation.

### Reload Documents

Click "🔄 Reload Documents" if you've added new files to the `documents/` folder manually.

---

## 🔧 Configuration

### Adjust Settings

The web app uses optimized settings by default. To customize, edit `app.py`:

```python
config = RAGConfig(
    chunk_size=1000,           # Size of text chunks
    chunk_overlap=200,         # Overlap between chunks
    embedding_model="...",     # Embedding model
    top_k_retrieval=5,         # Number of results
    # ... more settings
)
```

### Change Port

By default, Streamlit runs on port 8501. To change:

```bash
streamlit run app.py --server.port 8080
```

### Run on Network

To access from other devices on your network:

```bash
streamlit run app.py --server.address 0.0.0.0
```

---

## 📊 System Architecture

```
User Browser
     ↓
Streamlit Frontend (app.py)
     ↓
RAG Pipeline (src/rag_pipeline.py)
     ↓
├─ Document Processor
├─ Embedding Engine  
├─ Vector Database (ChromaDB)
└─ Retrieval Engine
     ↓
Display Results
```

---

## 💡 Tips for Best Results

### 1. Ask Specific Questions
✅ "What is backpropagation in neural networks?"
❌ "Tell me about AI"

### 2. Use Keywords from Documents
The system works best when your questions contain terms from the knowledge base.

### 3. Check Sources
Always review the source passages to understand the context of answers.

### 4. Upload Relevant Documents
Add your own materials for domain-specific questions.

### 5. Iterate Questions
If you don't get a good answer, rephrase your question.

---

## 🐛 Troubleshooting

### App Won't Start

**Error: "Streamlit not found"**
```bash
pip install streamlit>=1.28.0
```

**Error: "Port already in use"**
```bash
# Kill existing process
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

### No Documents Loaded

**Solution 1: Run setup**
```bash
python setup.py
```

**Solution 2: Check documents folder**
```bash
ls documents/
# Should show .txt files
```

**Solution 3: Reload in app**
Click "🔄 Reload Documents" in sidebar

### Slow Performance

**Solution 1: Use faster model**
Edit `app.py`, change embedding model to:
```python
embedding_model="sentence-transformers/all-MiniLM-L6-v2"
```

**Solution 2: Reduce chunks retrieved**
```python
top_k_retrieval=3  # Instead of 5
```

### Upload Not Working

**Check file format:** Only PDF and TXT supported
**Check file size:** Very large files may take time
**Check permissions:** Ensure app can write to `documents/uploaded/`

---

## 🎨 Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add More Sample Questions

Edit `app.py`, find the `sample_questions` list:

```python
sample_questions = [
    "Your custom question 1",
    "Your custom question 2",
    # ... add more
]
```

### Modify UI Layout

Edit `app.py` to customize:
- Column widths: `st.columns([2, 1])`
- Sidebar content
- Color scheme
- Display format

---

## 📈 Advanced Features

### Enable LLM Generation

To get AI-generated answers (not just retrieval):

1. Get OpenAI API key from https://platform.openai.com
2. Create `.env` file:
   ```
   OPENAI_API_KEY=your-key-here
   ```
3. Modify `app.py` to use full RAG with generation

### Add More Document Types

Edit `src/document_processor.py` to support:
- DOCX files (already supported)
- Markdown
- CSV
- JSON
- HTML

### Enable Reranking

1. Get Cohere API key from https://cohere.ai
2. Add to `.env`:
   ```
   COHERE_API_KEY=your-key-here
   ```
3. In `app.py`, set:
   ```python
   use_reranking=True
   ```

---

## 🚀 Deployment

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### Cloud Deployment

**Streamlit Cloud:**
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect your repo
4. Deploy!

**Docker:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

**Heroku, AWS, GCP:**
See Streamlit deployment docs

---

## 📞 Support

- Check console for error messages
- Review `TEST_GUIDE.md` for system tests
- See `README.md` for detailed documentation
- Visit Streamlit docs: https://docs.streamlit.io

---

## 🎉 Next Steps

1. **Explore the UI** - Click around and try features
2. **Ask questions** - Test with sample questions
3. **Upload documents** - Add your own materials
4. **Customize** - Modify to fit your needs
5. **Deploy** - Share with others!

Enjoy your RAG AI Assistant! 🤖✨

