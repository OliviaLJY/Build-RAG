"""
Script to ingest documents with the new intelligent configuration
"""

import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Load environment variables
load_dotenv()

print("🚀 Ingesting Documents with Intelligent Configuration...")
print("=" * 60)

# Create intelligent configuration
config = RAGConfig(
    llm_model="gpt-4o",
    embedding_model="text-embedding-3-large",
    temperature=0.3,
    max_tokens=4000,
    top_k_retrieval=5,
    use_hybrid_search=True,
    use_contextual_compression=True,
    collection_name='production_rag_documents'
)

print(f"📊 Using OpenAI Embeddings: {config.embedding_model}")
print(f"🧠 LLM Model: {config.llm_model}")

# Initialize RAG pipeline
rag = RAGPipeline(config)

# Ingest documents
print("\n📚 Ingesting documents from ./documents/ ...")
try:
    # This will ingest all documents including subdirectories
    rag.ingest_documents("documents", create_new=True)
    
    stats = rag.get_stats()
    print(f"\n✅ Success!")
    print(f"   📄 Total document chunks: {stats['num_documents']}")
    print(f"   📦 Chunk size: {stats['chunk_size']}")
    print(f"   🔄 Chunk overlap: {stats['chunk_overlap']}")
    print(f"   💾 Vector store: {stats['vector_store_type']}")
    print(f"   🔍 Hybrid search: {stats['hybrid_search_enabled']}")
    
    print("\n" + "=" * 60)
    print("✅ Documents ingested successfully with OpenAI embeddings!")
    print("   Now your RAG system can answer intelligent questions.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

