# RAG System - Testing Guide

## Quick Test (5 minutes)

### Step 1: Install Dependencies

```bash
# Navigate to the project directory
cd /Users/lijiayu/Desktop/RAG

# Install requirements
pip install -r requirements.txt
```

### Step 2: Run Setup

```bash
# Creates folders and sample document
python setup.py
```

Expected output:
```
====================================================
 RAG System Setup
====================================================

Creating directories...
✓ Created directory: documents
✓ Created directory: data
✓ Created directory: data/vectorstore
✓ Created directory: logs

Creating sample documents...
✓ Created sample document: documents/rag_introduction.txt

====================================================
 Setup Complete!
====================================================
```

### Step 3: Run the Test Script

```bash
# Run the automated test
python test_rag.py
```

This will test all components automatically.

## Manual Testing Steps

### Test 1: Basic Document Processing

```bash
python -c "
from src.document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
docs = processor.load_documents('documents/rag_introduction.txt')
print(f'✓ Loaded {len(docs)} documents')

chunks = processor.process_documents(docs)
print(f'✓ Created {len(chunks)} chunks')
print(f'✓ Sample chunk: {chunks[0].page_content[:100]}...')
"
```

**Expected**: Should load 1 document and create multiple chunks.

### Test 2: Embedding Generation

```bash
python -c "
from src.embeddings import EmbeddingManager

manager = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
print('✓ Embedding manager initialized')

embedding = manager.embed_query('What is RAG?')
print(f'✓ Generated embedding with {len(embedding)} dimensions')
"
```

**Expected**: Should create 384-dimensional embeddings (for MiniLM model).

### Test 3: Vector Store

```bash
python -c "
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from langchain.docstore.document import Document

# Initialize
embeddings = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = VectorStoreManager(
    embeddings=embeddings.get_embeddings_instance(),
    store_type='chromadb',
    collection_name='test_collection'
)

# Create sample docs
docs = [
    Document(page_content='Python is a programming language.'),
    Document(page_content='Machine learning uses algorithms.'),
    Document(page_content='RAG combines retrieval and generation.')
]

# Store and search
vector_store.create_vector_store(docs)
print(f'✓ Created vector store with {len(docs)} documents')

results = vector_store.similarity_search('What is RAG?', k=1)
print(f'✓ Search returned {len(results)} results')
print(f'✓ Top result: {results[0].page_content}')
"
```

**Expected**: Should find the RAG-related document.

### Test 4: Full RAG Pipeline (Without LLM)

```bash
python -c "
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Configure without LLM requirements
config = RAGConfig(
    chunk_size=500,
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    use_reranking=False,
    use_hybrid_search=False
)

# Initialize and ingest
rag = RAGPipeline(config)
print('✓ RAG pipeline initialized')

rag.ingest_documents('documents/rag_introduction.txt', create_new=True)
print('✓ Documents ingested')

# Test retrieval (without generation)
docs = rag.retrieve('What is RAG?')
print(f'✓ Retrieved {len(docs)} documents')
print(f'✓ Top document preview: {docs[0].page_content[:150]}...')
"
```

**Expected**: Should retrieve relevant documents about RAG.

### Test 5: Interactive Chat (Full System)

```bash
python examples/interactive_chat.py
```

**Test queries to try:**
1. "What is RAG?"
2. "How does retrieval work?"
3. "What are the advantages of RAG?"
4. "Explain the key algorithms"

**Expected**: Interactive chat session with answers from the sample document.

## Component-Specific Tests

### Test Chunking Strategies

```bash
python -c "
from src.document_processor import DocumentProcessor

text = 'This is a test. ' * 100  # 100 sentences

# Test recursive
proc1 = DocumentProcessor(chunk_size=200, chunking_strategy='recursive')
chunks1 = proc1.process_text(text)
print(f'✓ Recursive chunking: {len(chunks1)} chunks')

# Test token-based
proc2 = DocumentProcessor(chunk_size=200, chunking_strategy='token')
chunks2 = proc2.process_text(text)
print(f'✓ Token chunking: {len(chunks2)} chunks')
"
```

### Test Different Embedding Models

```bash
python -c "
from src.embeddings import EmbeddingManager
import time

models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

for model in models:
    start = time.time()
    manager = EmbeddingManager(model_name=model)
    embedding = manager.embed_query('test query')
    duration = time.time() - start
    print(f'✓ {model.split(\"/\")[1]}: {len(embedding)} dims, {duration:.2f}s')
"
```

### Test Vector Store Types

```bash
# Test ChromaDB
python -c "
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from langchain.docstore.document import Document

docs = [Document(page_content=f'Document {i}') for i in range(10)]

# ChromaDB
embeddings = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
vs_chroma = VectorStoreManager(
    embeddings=embeddings.get_embeddings_instance(),
    store_type='chromadb',
    collection_name='test_chroma'
)
vs_chroma.create_vector_store(docs)
print('✓ ChromaDB test passed')

# FAISS
vs_faiss = VectorStoreManager(
    embeddings=embeddings.get_embeddings_instance(),
    store_type='faiss',
    persist_directory='./data/test_faiss'
)
vs_faiss.create_vector_store(docs)
print('✓ FAISS test passed')
"
```

## Performance Testing

### Test Large Document Processing

```bash
python -c "
from src.document_processor import DocumentProcessor
import time

# Create large text
large_text = 'This is a sentence about artificial intelligence. ' * 1000

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

start = time.time()
chunks = processor.process_text(large_text)
duration = time.time() - start

print(f'✓ Processed {len(large_text)} characters')
print(f'✓ Created {len(chunks)} chunks')
print(f'✓ Time taken: {duration:.2f} seconds')
print(f'✓ Speed: {len(large_text)/duration:.0f} chars/sec')
"
```

### Test Search Performance

```bash
python -c "
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from langchain.docstore.document import Document
import time

# Create test documents
docs = [Document(page_content=f'Document about topic {i}') for i in range(100)]

embeddings = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = VectorStoreManager(
    embeddings=embeddings.get_embeddings_instance(),
    store_type='chromadb'
)

# Measure indexing time
start = time.time()
vector_store.create_vector_store(docs)
index_time = time.time() - start

# Measure search time
start = time.time()
results = vector_store.similarity_search('topic', k=5)
search_time = time.time() - start

print(f'✓ Indexed {len(docs)} documents in {index_time:.2f}s')
print(f'✓ Search took {search_time*1000:.2f}ms')
print(f'✓ Retrieved {len(results)} results')
"
```

## Troubleshooting Tests

### If embeddings fail:
```bash
# Test with smallest model
python -c "
from src.embeddings import EmbeddingManager
try:
    manager = EmbeddingManager(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print('✓ Embeddings working')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

### If out of memory:
```bash
# Test with minimal settings
python -c "
from src.config import RAGConfig
config = RAGConfig(
    chunk_size=300,
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    use_hybrid_search=False,
    use_compression=False
)
print(f'✓ Minimal config: chunk_size={config.chunk_size}')
"
```

### Check dependencies:
```bash
python -c "
import sys
packages = ['langchain', 'chromadb', 'sentence_transformers', 'faiss']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} installed')
    except ImportError:
        print(f'✗ {pkg} NOT installed')
"
```

## Expected Test Results Summary

| Test | Expected Result | Time |
|------|----------------|------|
| Setup | Creates folders and sample doc | < 1s |
| Document Processing | ~20-30 chunks from sample | < 1s |
| Embedding Generation | 384 or 768-dim vectors | 1-3s |
| Vector Store Creation | Success, no errors | 2-5s |
| Similarity Search | Returns relevant docs | < 0.1s |
| Full Pipeline | Retrieves 3-5 docs | 3-10s |
| Interactive Chat | Answers questions | 5-15s/query |

## Success Criteria

✅ All imports work without errors  
✅ Documents load and chunk properly  
✅ Embeddings generate correct dimensions  
✅ Vector store creates and persists  
✅ Search returns relevant results  
✅ Pipeline completes end-to-end  

## Next Steps After Testing

1. **Add your own documents**: Place files in `documents/` folder
2. **Customize configuration**: Edit settings in `src/config.py`
3. **Try advanced examples**: Run `examples/advanced_usage.py`
4. **Set up API keys**: Add `.env` file for OpenAI/Cohere (optional)

## Getting Help

If tests fail:
1. Check the error message carefully
2. Verify all dependencies installed: `pip list`
3. Try with minimal configuration
4. Check `TEST_GUIDE.md` troubleshooting section
5. Review `README.md` for detailed setup

