"""
Advanced usage example with custom configuration and features
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager


def example_custom_document_processing():
    """Example of custom document processing"""
    print("="*50)
    print("Custom Document Processing")
    print("="*50)
    
    # Create processor with semantic chunking
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100,
        chunking_strategy="recursive"  # Options: 'recursive', 'token', 'semantic'
    )
    
    # Process sample text
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a powerful technique that combines 
    information retrieval with text generation. It works by first retrieving 
    relevant documents from a knowledge base, then using those documents as 
    context for a language model to generate accurate, informed responses.
    
    The key advantages of RAG include: reduced hallucinations, ability to work 
    with up-to-date information, better factual accuracy, and transparency 
    through source attribution.
    """
    
    documents = processor.process_text(
        sample_text,
        metadata={"source": "example", "topic": "RAG"}
    )
    
    print(f"\nCreated {len(documents)} chunks from sample text")
    for i, doc in enumerate(documents, 1):
        print(f"\nChunk {i}:")
        print(f"  Content: {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")


def example_similarity_search():
    """Example of direct similarity search"""
    print("\n" + "="*50)
    print("Direct Similarity Search")
    print("="*50)
    
    config = RAGConfig()
    
    # Initialize components
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model
    )
    
    vector_store = VectorStoreManager(
        embeddings=embedding_manager.get_embeddings_instance(),
        store_type="chromadb",
        collection_name="test_collection"
    )
    
    # Create sample documents
    from langchain.docstore.document import Document
    
    sample_docs = [
        Document(page_content="Python is a high-level programming language.", metadata={"id": 1}),
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"id": 2}),
        Document(page_content="RAG combines retrieval with generation for better AI responses.", metadata={"id": 3}),
        Document(page_content="Vector databases store embeddings for semantic search.", metadata={"id": 4}),
    ]
    
    # Create vector store
    vector_store.create_vector_store(sample_docs)
    
    # Perform search
    query = "What is RAG?"
    results = vector_store.similarity_search_with_score(query, k=2)
    
    print(f"\nQuery: {query}")
    print("\nResults:")
    for doc, score in results:
        print(f"\n  Score: {score:.4f}")
        print(f"  Content: {doc.page_content}")
        print(f"  Metadata: {doc.metadata}")


def example_incremental_updates():
    """Example of adding documents incrementally"""
    print("\n" + "="*50)
    print("Incremental Document Updates")
    print("="*50)
    
    config = RAGConfig(
        collection_name="incremental_demo"
    )
    
    rag = RAGPipeline(config)
    
    # Initial ingestion
    print("\n1. Creating initial document store...")
    processor = DocumentProcessor()
    
    initial_text = "Initial document about artificial intelligence and machine learning."
    initial_docs = processor.process_text(initial_text, metadata={"batch": 1})
    
    # We'll create the vector store manually for this example
    from langchain.docstore.document import Document
    rag.documents = [Document(page_content=initial_text, metadata={"batch": 1})]
    rag.vector_store_manager.create_vector_store(rag.documents)
    rag._initialize_retriever()
    
    # Add more documents
    print("2. Adding new documents...")
    new_text = "Additional information about natural language processing and transformers."
    new_docs = processor.process_text(new_text, metadata={"batch": 2})
    
    rag.documents.extend(new_docs)
    rag.vector_store_manager.add_documents(new_docs)
    
    print(f"\nTotal documents: {len(rag.documents)}")
    
    # Test query
    result = rag.query("What topics are covered?", return_source_documents=True)
    print(f"\nAnswer: {result['answer']}")


def example_metadata_filtering():
    """Example of filtering by metadata"""
    print("\n" + "="*50)
    print("Metadata Filtering")
    print("="*50)
    
    config = RAGConfig()
    
    embedding_manager = EmbeddingManager(model_name=config.embedding_model)
    vector_store = VectorStoreManager(
        embeddings=embedding_manager.get_embeddings_instance(),
        collection_name="filtered_search"
    )
    
    # Create documents with different categories
    from langchain.docstore.document import Document
    
    docs = [
        Document(page_content="Python programming basics", metadata={"category": "programming"}),
        Document(page_content="Machine learning with Python", metadata={"category": "ml"}),
        Document(page_content="Data science fundamentals", metadata={"category": "data"}),
        Document(page_content="Deep learning neural networks", metadata={"category": "ml"}),
    ]
    
    vector_store.create_vector_store(docs)
    
    # Search with filter
    query = "learning"
    filter_dict = {"category": "ml"}
    
    results = vector_store.similarity_search(query, k=2, filter_dict=filter_dict)
    
    print(f"\nQuery: {query}")
    print(f"Filter: {filter_dict}")
    print("\nResults:")
    for doc in results:
        print(f"  - {doc.page_content} | Category: {doc.metadata['category']}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print(" RAG System - Advanced Examples")
    print("="*60)
    
    try:
        example_custom_document_processing()
        example_similarity_search()
        example_incremental_updates()
        example_metadata_filtering()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all requirements are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()

