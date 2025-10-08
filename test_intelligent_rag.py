"""
Test script to demonstrate the enhanced intelligent RAG system with GPT-4o
"""

import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Load environment variables
load_dotenv()

# Verify API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment")
    print("Please make sure your .env file exists and contains the API key")
    exit(1)

print("✅ OpenAI API key loaded successfully")
print(f"   Key preview: {api_key[:15]}...")

# Create enhanced configuration
print("\n🚀 Initializing Intelligent RAG System...")
print("=" * 60)

config = RAGConfig(
    # OpenAI GPT-4o - Most intelligent model
    llm_model="gpt-4o",
    
    # OpenAI embeddings for better semantic understanding
    embedding_model="text-embedding-3-large",
    
    # Optimized for intelligence
    temperature=0.3,  # Lower = more focused and accurate
    max_tokens=4000,  # Longer, more detailed responses
    
    # Advanced retrieval features
    top_k_retrieval=5,
    use_hybrid_search=True,
    use_contextual_compression=True,
    
    # Storage
    collection_name='production_rag_documents'  # Same as ingestion
)

print(f"📊 Configuration:")
print(f"   LLM Model: {config.llm_model} (Latest & Most Intelligent)")
print(f"   Embedding Model: {config.embedding_model}")
print(f"   Temperature: {config.temperature} (Focused & Accurate)")
print(f"   Max Tokens: {config.max_tokens} (Detailed Responses)")
print(f"   Hybrid Search: {config.use_hybrid_search}")
print(f"   Contextual Compression: {config.use_contextual_compression}")

# Initialize RAG pipeline
print("\n📚 Initializing RAG Pipeline...")
rag = RAGPipeline(config)

# Load existing documents
try:
    print("📖 Loading existing vector store...")
    rag.load_existing_store()
    print("✅ Vector store loaded successfully")
except Exception as e:
    print(f"⚠️  Could not load existing store: {e}")
    print("📁 Ingesting documents from ./documents/...")
    try:
        rag.ingest_documents("documents", create_new=True)
        print("✅ Documents ingested successfully")
    except Exception as e2:
        print(f"❌ Error ingesting documents: {e2}")
        exit(1)

# Get stats
stats = rag.get_stats()
print(f"\n📈 System Stats:")
print(f"   Documents: {stats['num_documents']} chunks")
print(f"   Chunk Size: {stats['chunk_size']}")
print(f"   Vector Store: {stats['vector_store_type']}")

# Test with intelligent queries
print("\n" + "=" * 60)
print("🧠 Testing Intelligent RAG System")
print("=" * 60)

test_questions = [
    "What is RAG and how does it work?",
    "Explain the architecture of neural networks",
    "What are the key differences between supervised and unsupervised learning?",
]

for i, question in enumerate(test_questions, 1):
    print(f"\n❓ Question {i}: {question}")
    print("-" * 60)
    
    try:
        result = rag.query(question, return_source_documents=True)
        
        print(f"💡 Answer:\n{result['answer']}")
        
        if result.get('source_documents'):
            print(f"\n📚 Sources ({len(result['source_documents'])} documents):")
            for j, doc in enumerate(result['source_documents'][:2], 1):
                preview = doc.page_content[:150].replace('\n', ' ')
                print(f"   {j}. {preview}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 60)

print("\n" + "=" * 60)
print("✅ Testing Complete!")
print("=" * 60)
print("\n💡 Your RAG system is now powered by GPT-4o for maximum intelligence!")
print("   - More accurate answers")
print("   - Better reasoning and context understanding")
print("   - Improved semantic search with OpenAI embeddings")

