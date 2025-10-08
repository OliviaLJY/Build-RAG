"""
Test external knowledge capability after enabling the fix
"""

from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

load_dotenv()

print("=" * 70)
print("🧪 Testing External Knowledge Feature")
print("=" * 70)

# Test with general knowledge ENABLED (default)
print("\n📊 MODE 1: General Knowledge ENABLED (allow_general_knowledge=True)")
print("-" * 70)

config_enabled = RAGConfig(
    collection_name='production_rag_documents',
    allow_general_knowledge=True  # NEW FEATURE!
)

rag_enabled = RAGPipeline(config_enabled)
rag_enabled.load_existing_store()

print("\n✅ Configuration:")
print(f"   allow_general_knowledge: {config_enabled.allow_general_knowledge}")
print(f"   llm_model: {config_enabled.llm_model}")

# Test 1: Question IN your documents
print("\n" + "=" * 70)
print("TEST 1: Question about content IN your documents")
print("=" * 70)
print("❓ Question: What is RAG?")
print("\nProcessing...\n")

result1 = rag_enabled.query("What is RAG?", return_source_documents=True)
print(f"💡 Answer preview: {result1['answer'][:150]}...")
print(f"📚 Sources: {len(result1['source_documents'])} documents")
print(f"🔍 Knowledge source: {result1.get('knowledge_source', 'documents')}")

# Test 2: Question NOT in your documents - NOW IT SHOULD ANSWER!
print("\n" + "=" * 70)
print("TEST 2: Question NOT in your documents (NEW BEHAVIOR)")
print("=" * 70)
print("❓ Question: Who is the president of France in 2024?")
print("\nProcessing...\n")

result2 = rag_enabled.query("Who is the president of France in 2024?", return_source_documents=True)
print(f"💡 Answer:\n{result2['answer']}")
print(f"📚 Sources: {len(result2['source_documents'])} documents")
print(f"🔍 Knowledge source: {result2.get('knowledge_source', 'unknown')}")

# Test 3: Another external knowledge question
print("\n" + "=" * 70)
print("TEST 3: General knowledge question")
print("=" * 70)
print("❓ Question: What is the capital of Japan?")
print("\nProcessing...\n")

result3 = rag_enabled.query("What is the capital of Japan?", return_source_documents=True)
print(f"💡 Answer:\n{result3['answer']}")
print(f"📚 Sources: {len(result3['source_documents'])} documents")
print(f"🔍 Knowledge source: {result3.get('knowledge_source', 'unknown')}")

# Test with general knowledge DISABLED (strict mode)
print("\n\n" + "=" * 70)
print("📊 MODE 2: General Knowledge DISABLED (allow_general_knowledge=False)")
print("-" * 70)

config_disabled = RAGConfig(
    collection_name='production_rag_documents',
    allow_general_knowledge=False  # STRICT MODE
)

rag_disabled = RAGPipeline(config_disabled)
rag_disabled.load_existing_store()

print("\n✅ Configuration:")
print(f"   allow_general_knowledge: {config_disabled.allow_general_knowledge}")
print(f"   llm_model: {config_disabled.llm_model}")

# Test 4: Same question with strict mode
print("\n" + "=" * 70)
print("TEST 4: External question in STRICT mode (old behavior)")
print("=" * 70)
print("❓ Question: Who is the president of France in 2024?")
print("\nProcessing...\n")

result4 = rag_disabled.query("Who is the president of France in 2024?", return_source_documents=True)
print(f"💡 Answer:\n{result4['answer']}")
print(f"📚 Sources: {len(result4['source_documents'])} documents")
print(f"🔍 Knowledge source: {result4.get('knowledge_source', 'unknown')}")

# Summary
print("\n\n" + "=" * 70)
print("📊 FEATURE COMPARISON")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│                    allow_general_knowledge=True                 │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Questions in documents: Answered with document context       │
│ ✅ Questions NOT in documents: Answered with general knowledge  │
│ ✅ Clear disclaimer when using general knowledge                │
│ ✅ More useful as general-purpose assistant                     │
│ ⚠️  May answer with outdated info (GPT-4o training cutoff)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    allow_general_knowledge=False                │
├─────────────────────────────────────────────────────────────────┤
│ ✅ Questions in documents: Answered with document context       │
│ ❌ Questions NOT in documents: "I couldn't find..."            │
│ ✅ Never hallucinates (strict document-grounding)               │
│ ✅ All answers are traceable to sources                         │
│ ⚠️  Less useful for general questions                           │
└─────────────────────────────────────────────────────────────────┘

🎯 DEFAULT: allow_general_knowledge=True (enabled)
   - More flexible and user-friendly
   - Clearly marks when using general knowledge
   - Best for general-purpose AI assistant

🔒 Use False when:
   - You need strict source verification
   - Only want answers from YOUR documents
   - Working with sensitive/legal/medical content
""")

print("=" * 70)
print("✅ Test Complete!")
print("=" * 70)
print("\n💡 To change the setting:")
print("   config = RAGConfig(allow_general_knowledge=True)  # or False")

