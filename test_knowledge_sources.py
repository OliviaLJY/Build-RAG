"""
Test to demonstrate where knowledge comes from:
1. Your documents (RAG retrieval)
2. GPT-4o's built-in knowledge
"""

from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

load_dotenv()

print("=" * 70)
print("🧪 Testing Knowledge Sources")
print("=" * 70)

# Initialize RAG with your documents
config = RAGConfig(collection_name='production_rag_documents')
rag = RAGPipeline(config)
rag.load_existing_store()

print(f"\n📚 System loaded with {len(rag.documents)} document chunks")
print("=" * 70)

# Test 1: Question about content IN your documents
print("\n" + "="*70)
print("TEST 1: Question ABOUT content in your documents")
print("="*70)
print("❓ Question: What is RAG and how does it work?")
print("\n📖 This content EXISTS in your documents/rag_introduction.txt")
print("\nProcessing...\n")

result1 = rag.query("What is RAG and how does it work?", return_source_documents=True)
print("💡 Answer:")
print(result1['answer'])
print(f"\n📚 Sources: {len(result1['source_documents'])} documents retrieved from YOUR files")
for i, doc in enumerate(result1['source_documents'][:2], 1):
    print(f"\n   Source {i}: {doc.page_content[:150]}...")

# Test 2: Question NOT in your documents but GPT-4o knows
print("\n\n" + "="*70)
print("TEST 2: Question about something NOT in your documents")
print("="*70)
print("❓ Question: Who is the president of France in 2024?")
print("\n📖 This content DOES NOT exist in your documents")
print("   (But GPT-4o knows from its training data)")
print("\nProcessing...\n")

result2 = rag.query("Who is the president of France in 2024?", return_source_documents=True)
print("💡 Answer:")
print(result2['answer'])
print(f"\n📚 Sources: {len(result2['source_documents'])} documents retrieved")
print("   Note: Documents retrieved but probably not relevant")
print("   GPT-4o will use its built-in knowledge instead")

# Test 3: Hybrid - uses both
print("\n\n" + "="*70)
print("TEST 3: Question that uses BOTH sources")
print("="*70)
print("❓ Question: How is RAG different from traditional chatbots?")
print("\n📖 Your documents explain RAG")
print("   GPT-4o knows about traditional chatbots")
print("   Answer will combine BOTH sources")
print("\nProcessing...\n")

result3 = rag.query("How is RAG different from traditional chatbots?", return_source_documents=True)
print("💡 Answer:")
print(result3['answer'])
print(f"\n📚 Sources: {len(result3['source_documents'])} documents from YOUR files")

# Summary
print("\n\n" + "="*70)
print("📊 SUMMARY: Where Knowledge Comes From")
print("="*70)
print("""
1. ✅ PRIMARY: Your Documents
   - RAG retrieves relevant chunks from your 87 documents
   - This is your specific, custom knowledge base
   - Retrieved based on semantic similarity to the query

2. ✅ SECONDARY: GPT-4o's Training
   - GPT-4o was trained on vast internet data (up to ~2023)
   - Has general world knowledge
   - Used to understand context and generate fluent answers

3. 🎯 COMBINATION:
   - RAG finds relevant info from YOUR documents
   - GPT-4o uses that info + its knowledge to generate answers
   - Prioritizes YOUR documents when available
   - Falls back to its training for general knowledge

4. 💡 IMPORTANT:
   - Without retrieval: GPT-4o only uses its training (may hallucinate)
   - With RAG: GPT-4o grounds answers in YOUR documents (more accurate)
   - Best of both worlds: Specific knowledge + intelligent generation
""")

print("="*70)
print("✅ Test Complete!")
print("="*70)

