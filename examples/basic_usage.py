"""
Basic usage example of the RAG system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig


def main():
    """Basic RAG usage example"""
    
    # Initialize configuration
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        top_k_retrieval=5,
        use_hybrid_search=True,
        use_reranking=False  # Set to True if you have Cohere API key
    )
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(config)
    
    # Ingest documents
    # Replace with your document path
    documents_path = "./documents"  # Can be a file or directory
    
    print(f"\nIngesting documents from: {documents_path}")
    print("Note: Make sure to place your documents in the 'documents' folder")
    
    try:
        rag.ingest_documents(documents_path, create_new=True)
        
        # Print stats
        print("\nRAG System Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example queries
        queries = [
            "What is the main topic of these documents?",
            "Summarize the key points",
            "What are the important findings?"
        ]
        
        print("\n" + "="*50)
        print("Example Queries")
        print("="*50)
        
        for query in queries:
            print(f"\nQuestion: {query}")
            print("-" * 50)
            
            result = rag.query(query, return_source_documents=True)
            print(f"Answer: {result['answer']}")
            
            if result.get('source_documents'):
                print(f"\nSources ({len(result['source_documents'])} documents):")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    print(f"  {i}. {doc.page_content[:100]}...")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Create a 'documents' folder with your documents")
        print("2. Install all requirements: pip install -r requirements.txt")
        print("3. Set up your API keys if using OpenAI/Cohere")


if __name__ == "__main__":
    main()

