"""
Quick interactive chat using the already-cached smaller model
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig


def interactive_chat():
    """Run interactive chat session with fast model"""
    
    print("="*60)
    print(" RAG Quick Chat (using fast cached model)")
    print("="*60)
    
    # Use the already-downloaded smaller/faster model
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Fast, already cached!
        top_k_retrieval=5,
        use_hybrid_search=False,  # Simplified for speed
        use_reranking=False,
        use_contextual_compression=False
    )
    
    rag = RAGPipeline(config)
    
    # Always process the sample document for this demo
    print("\nProcessing documents...")
    documents_path = "documents/rag_introduction.txt"
    
    if not Path(documents_path).exists():
        print("Sample document not found. Run: python setup.py")
        return
    
    rag.ingest_documents(documents_path, create_new=True)
    print("Documents processed!")
    
    # Display stats
    print("\n" + "-"*60)
    print("RAG System Stats:")
    stats = rag.get_stats()
    print(f"  Documents: {stats.get('num_documents', 0)} chunks")
    print(f"  Model: {stats.get('embedding_model', 'N/A').split('/')[-1]}")
    print("-"*60)
    
    # Start chat loop
    print("\n💬 Chat started! Commands:")
    print("  'quit' or 'exit' - End session")
    print("  'stats' - Show system stats")
    print("  'sources' - Toggle source display\n")
    
    show_sources = True
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 You: ").strip()
            
            if not question:
                continue
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                print("\nSystem Statistics:")
                stats = rag.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if question.lower() == 'sources':
                show_sources = not show_sources
                print(f"\n📚 Source display: {'ON' if show_sources else 'OFF'}")
                continue
            
            # Get answer (retrieval only, no LLM generation)
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Just retrieve documents (no OpenAI needed)
            docs = rag.retrieve(question)
            
            if not docs:
                print("I couldn't find relevant information about that.")
                continue
            
            # Show the most relevant passage
            print(f"\n{docs[0].page_content}\n")
            
            # Show sources if enabled
            if show_sources and len(docs) > 1:
                print(f"📚 Related information ({len(docs)-1} more passages):")
                for i, doc in enumerate(docs[1:], 1):
                    preview = doc.page_content[:80].replace('\n', ' ')
                    print(f"  [{i}] {preview}...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def main():
    """Main entry point"""
    try:
        interactive_chat()
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

