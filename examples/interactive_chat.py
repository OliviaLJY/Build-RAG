"""
Interactive chat interface for the RAG system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig


def interactive_chat():
    """Run interactive chat session"""
    
    print("="*60)
    print(" RAG Interactive Chat")
    print("="*60)
    
    # Initialize RAG
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        top_k_retrieval=5,
        use_hybrid_search=True
    )
    
    rag = RAGPipeline(config)
    
    # Check if vector store exists
    store_path = Path(config.persist_directory)
    
    if store_path.exists():
        print("\nFound existing vector store. Loading...")
        try:
            rag.load_existing_store()
            print("Vector store loaded successfully!")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Please ingest documents first using basic_usage.py")
            return
    else:
        print("\nNo existing vector store found.")
        documents_path = input("Enter path to documents (or press Enter to exit): ").strip()
        
        if not documents_path:
            print("Exiting...")
            return
        
        print("\nIngesting documents...")
        try:
            rag.ingest_documents(documents_path, create_new=True)
            print("Documents ingested successfully!")
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return
    
    # Display stats
    print("\n" + "-"*60)
    print("RAG System Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("-"*60)
    
    # Start chat loop
    print("\n💬 Chat started! Type 'quit' or 'exit' to end the session.")
    print("Type 'stats' to see system statistics.")
    print("Type 'sources' to toggle source document display.\n")
    
    show_sources = True
    
    while True:
        try:
            # Get user input
            question = input("\n🤔 You: ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == 'stats':
                print("\nRAG System Statistics:")
                stats = rag.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if question.lower() == 'sources':
                show_sources = not show_sources
                print(f"\nSource display: {'ON' if show_sources else 'OFF'}")
                continue
            
            # Get answer
            print("\n🤖 Assistant: ", end="", flush=True)
            
            result = rag.query(question, return_source_documents=show_sources)
            print(result['answer'])
            
            # Show sources if enabled
            if show_sources and result.get('source_documents'):
                print(f"\n📚 Sources ({len(result['source_documents'])} documents):")
                for i, doc in enumerate(result['source_documents'], 1):
                    # Show preview of source
                    preview = doc.page_content[:150].replace('\n', ' ')
                    print(f"  [{i}] {preview}...")
                    
                    # Show metadata if available
                    if doc.metadata:
                        metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                        print(f"      Metadata: {metadata_str}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """Main entry point"""
    try:
        interactive_chat()
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

