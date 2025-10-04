# Quick Start Guide

Get up and running with the RAG system in 5 minutes!

## Step 1: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Step 2: Setup

Run the setup script to create necessary directories and sample documents:

```bash
python setup.py
```

This creates:
- `documents/` - Place your documents here
- `data/` - Vector store persistence
- `documents/rag_introduction.txt` - Sample document

## Step 3: Run Your First Query

### Option A: Interactive Chat (Recommended)

```bash
python examples/interactive_chat.py
```

This will:
1. Process the sample document
2. Start an interactive chat session
3. Let you ask questions about the content

### Option B: Basic Script

```bash
python examples/basic_usage.py
```

## Step 4: Use Your Own Documents

1. **Add your documents**:
   ```bash
   # Place PDFs, TXT, or DOCX files in documents/
   cp /path/to/your/docs/* documents/
   ```

2. **Process them**:
   ```python
   from src.rag_pipeline import RAGPipeline
   
   rag = RAGPipeline()
   rag.ingest_documents("./documents", create_new=True)
   ```

3. **Query**:
   ```python
   answer = rag.chat("Your question here")
   print(answer)
   ```

## Configuration Options

Edit settings by creating a config object:

```python
from src.config import RAGConfig

config = RAGConfig(
    chunk_size=1000,              # Adjust chunk size
    chunk_overlap=200,            # Overlap between chunks
    top_k_retrieval=5,            # Number of documents to retrieve
    use_hybrid_search=True,       # Enable hybrid search
    use_reranking=False,          # Enable if you have Cohere API
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

## Optional: API Keys for Enhanced Features

Create a `.env` file for API keys:

```env
# For OpenAI LLM (if you want to use GPT models)
OPENAI_API_KEY=sk-your-key-here

# For Cohere Reranking (optional, improves results)
COHERE_API_KEY=your-cohere-key-here
```

**Without API Keys**: The system still works! It will:
- Use local HuggingFace embeddings (free)
- Skip LLM generation (you can just retrieve documents)
- Disable reranking

## Common Use Cases

### 1. Simple Q&A
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.load_existing_store()  # Load previously processed docs
answer = rag.chat("What is RAG?")
print(answer)
```

### 2. Get Source Documents
```python
result = rag.query("Your question", return_source_documents=True)
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for doc in result['source_documents']:
    print(f"- {doc.page_content[:100]}...")
```

### 3. Add More Documents Later
```python
# Initial documents
rag.ingest_documents("./documents/batch1", create_new=True)

# Add more later without recreating
rag.ingest_documents("./documents/batch2", create_new=False)
```

### 4. Search Without Generation
```python
# Just retrieve relevant documents
documents = rag.retrieve("search query")
for doc in documents:
    print(doc.page_content)
```

## Performance Tips

### Fast Setup (Good for Testing)
```python
config = RAGConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster
    chunk_size=500,
    top_k_retrieval=3,
    use_hybrid_search=False,
    use_compression=False
)
```

### Best Quality Setup
```python
config = RAGConfig(
    embedding_model="sentence-transformers/all-mpnet-base-v2",  # Better quality
    chunk_size=1000,
    top_k_retrieval=10,
    rerank_top_k=5,
    use_hybrid_search=True,
    use_reranking=True,  # Requires Cohere API key
    use_compression=True
)
```

### Large Documents
```python
config = RAGConfig(
    chunk_size=1500,           # Larger chunks
    chunk_overlap=300,         # More overlap
    vector_store_type="faiss"  # Faster for large scale
)
```

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "No documents found"
Make sure documents are in the `documents/` folder:
```bash
ls documents/
```

### Out of Memory
Reduce chunk size or use smaller embedding model:
```python
config = RAGConfig(
    chunk_size=500,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Slow Performance
Use FAISS instead of ChromaDB:
```python
config = RAGConfig(
    vector_store_type="faiss"
)
```

### Poor Results
1. Increase top_k: `top_k_retrieval=10`
2. Enable hybrid search: `use_hybrid_search=True`
3. Adjust chunk size based on your documents
4. Enable reranking if available

## Next Steps

1. **Read the full README**: `README.md` for detailed documentation
2. **Algorithm details**: `ALGORITHMS.md` for technical deep-dive
3. **Advanced examples**: Check `examples/advanced_usage.py`
4. **Customize**: Modify the code in `src/` for your needs

## Example Output

After running the interactive chat:

```
💬 Chat started! Type 'quit' or 'exit' to end the session.

🤔 You: What is RAG?

🤖 Assistant: Retrieval-Augmented Generation (RAG) is an advanced AI 
technique that enhances large language models by combining them with 
external knowledge retrieval. Instead of relying solely on training data, 
RAG systems retrieve relevant information from a knowledge base and use 
it as context for generating responses.

📚 Sources (3 documents):
  [1] Retrieval-Augmented Generation (RAG) is an advanced AI technique...
  [2] How RAG Works: The system processes and stores documents...
  [3] Key advantages of RAG include reduced hallucinations...
```

## Support

- Check the examples in `/examples`
- Read troubleshooting in `README.md`
- Review algorithm details in `ALGORITHMS.md`

Happy RAG building! 🚀

