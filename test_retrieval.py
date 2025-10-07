#!/usr/bin/env python3
"""
Test RAG Retrieval - Diagnose why documents aren't being retrieved
"""

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig
from pathlib import Path

print("🔍 RAG Retrieval Diagnostic Test")
print("=" * 70)

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

print("\n1️⃣ Initializing RAG Pipeline...")
pipeline = RAGPipeline(config)

print("\n2️⃣ Loading vector store...")
try:
    pipeline.load_existing_store()
    print("✅ Vector store loaded")
except Exception as e:
    print(f"❌ Error loading vector store: {e}")
    print("\nTrying to ingest documents instead...")
    pipeline.ingest_documents("documents", create_new=True)
    print("✅ Documents ingested")

print("\n3️⃣ Checking document count...")
stats = pipeline.get_stats()
print(f"   Number of documents in store: {stats['num_documents']}")

if stats['num_documents'] == 0:
    print("\n❌ ERROR: No documents in vector store!")
    print("   Run: python refresh_documents.py")
    exit(1)

print("\n4️⃣ Testing retrieval with different queries...")
print("-" * 70)

test_queries = [
    "What is machine learning?",
    "machine learning",
    "supervised learning",
    "neural networks",
    "deep learning"
]

for query in test_queries:
    print(f"\n📝 Query: '{query}'")
    try:
        documents = pipeline.retrieve(query)
        print(f"   ✅ Retrieved {len(documents)} documents")
        
        if documents:
            print(f"   Top result (first 200 chars):")
            print(f"   {documents[0].page_content[:200]}...")
            print(f"   Score/Relevance: {documents[0].metadata if hasattr(documents[0], 'metadata') else 'N/A'}")
        else:
            print(f"   ⚠️  No documents retrieved for this query")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("\n5️⃣ Direct vector store test...")

# Test directly accessing vector store
try:
    vectorstore = pipeline.vector_store_manager.vector_store
    
    # Try similarity search directly
    print("\n   Testing direct similarity search...")
    results = vectorstore.similarity_search("machine learning", k=3)
    print(f"   Direct search returned {len(results)} results")
    
    if results:
        print(f"\n   First result:")
        print(f"   {results[0].page_content[:300]}...")
    else:
        print("   ⚠️  Direct search returned no results")
        
except Exception as e:
    print(f"   ❌ Error in direct search: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("\n6️⃣ Testing with production API simulation...")

try:
    # Simulate what the API does
    question = "What is machine learning?"
    print(f"\n   Query: {question}")
    
    documents = pipeline.retrieve(question)
    
    if not documents:
        answer = "I couldn't find relevant information to answer your question."
        print(f"   ❌ Result: {answer}")
        print("\n   This is the problem! No documents retrieved.")
    else:
        answer = f"Based on the documents: {documents[0].page_content}"
        print(f"   ✅ Retrieved {len(documents)} documents")
        print(f"   Answer preview: {answer[:200]}...")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("\n📊 Diagnosis Summary:")
print("=" * 70)

if stats['num_documents'] > 0:
    print(f"✅ Documents loaded: {stats['num_documents']} chunks")
else:
    print(f"❌ No documents in vector store")

print("\n💡 Recommendations:")
print("   1. If no documents retrieved, try:")
print("      - Run: python refresh_documents.py")
print("      - Check embedding model compatibility")
print("   2. If documents exist but not retrieved:")
print("      - Check query phrasing")
print("      - Try more specific queries")
print("   3. Restart API server after fixing")
print("\n")

