#!/usr/bin/env python3
"""
Refresh Documents Script
Run this whenever you add/update documents to reload them into the RAG system
"""

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig
import shutil
from pathlib import Path

print("🔄 Refreshing RAG Document Store...")
print("=" * 60)

# Delete old vectorstore
vectorstore_path = Path("./data/vectorstore")
if vectorstore_path.exists():
    print(f"🗑️  Deleting old vectorstore: {vectorstore_path}")
    shutil.rmtree(vectorstore_path)
    print("✅ Old vectorstore deleted")
else:
    print("ℹ️  No existing vectorstore found")

print()

# Create pipeline with same config as production
config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k_retrieval=5,
    use_hybrid_search=False,
    use_reranking=False,
    collection_name='production_rag_documents'
)

print("📚 Initializing RAG Pipeline...")
pipeline = RAGPipeline(config)

# Ingest documents
doc_path = Path("documents")
if not doc_path.exists():
    print("❌ Error: 'documents' folder not found!")
    exit(1)

doc_files = list(doc_path.glob("*"))
if not doc_files:
    print("⚠️  Warning: No files found in 'documents' folder!")
    exit(1)

print(f"📄 Found {len(doc_files)} file(s) in documents/:")
for f in doc_files:
    print(f"   - {f.name}")

print()
print("🔨 Ingesting documents (this may take a minute)...")

try:
    pipeline.ingest_documents("documents", create_new=True)
    print()
    print("=" * 60)
    print("✅ Documents successfully ingested!")
    print("=" * 60)
    
    # Test retrieval
    print()
    print("🧪 Testing retrieval...")
    test_query = "What is machine learning?"
    docs = pipeline.retrieve(test_query)
    
    if docs:
        print(f"✅ Successfully retrieved {len(docs)} documents for test query!")
        print(f"   Query: '{test_query}'")
        print(f"   First result preview: {docs[0].page_content[:150]}...")
    else:
        print("⚠️  Warning: Test query returned no results")
    
    print()
    print("🎉 All done! Restart your server to use the new documents.")
    print("   Run: uvicorn api_server_production:app --reload")
    
except Exception as e:
    print(f"❌ Error during ingestion: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

