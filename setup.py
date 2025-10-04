"""
Setup script for RAG system
"""

from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        "documents",
        "data",
        "data/vectorstore",
        "logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_sample_documents():
    """Create sample documents for testing"""
    documents_dir = Path("documents")
    
    sample_content = """
Retrieval-Augmented Generation (RAG) Overview

What is RAG?
Retrieval-Augmented Generation (RAG) is an advanced AI technique that enhances large language models 
by combining them with external knowledge retrieval. Instead of relying solely on the model's training 
data, RAG systems retrieve relevant information from a knowledge base and use it as context for 
generating responses.

How RAG Works

1. Document Ingestion
The system processes and stores documents in a vector database. Documents are:
- Split into manageable chunks
- Converted to numerical embeddings using embedding models
- Stored in a vector store for efficient retrieval

2. Query Processing
When a user asks a question:
- The query is converted to an embedding
- Similar documents are retrieved using semantic search
- Optionally, keyword-based search (BM25) is combined for hybrid retrieval
- Retrieved documents are optionally reranked for better relevance

3. Answer Generation
The language model receives:
- The user's question
- Retrieved context from relevant documents
- A prompt template for structured responses

Key Algorithms in RAG Systems

Embedding Models
- Sentence Transformers: Open-source models for semantic similarity
- OpenAI Embeddings: High-quality commercial embeddings
- all-mpnet-base-v2: Best balance of quality and speed

Retrieval Algorithms
- Cosine Similarity: Measures semantic similarity between vectors
- BM25: Probabilistic keyword-based ranking algorithm
- Hybrid Search: Combines semantic and keyword approaches

Advanced Techniques
- Contextual Compression: Filters irrelevant content from retrieved documents
- Reranking: Re-scores documents using specialized models
- Metadata Filtering: Retrieves documents based on attributes

Advantages of RAG

1. Reduced Hallucinations
By grounding responses in retrieved documents, RAG significantly reduces the chance of the model 
generating false information.

2. Up-to-date Information
Unlike static training data, RAG can access current information by retrieving from updated 
knowledge bases.

3. Source Attribution
RAG systems can provide sources for their answers, increasing transparency and trustworthiness.

4. Domain Specialization
Organizations can create specialized RAG systems using their own documents without retraining 
large models.

5. Cost Effective
RAG avoids the need for expensive model fine-tuning while achieving specialized performance.

Best Practices

Document Processing
- Use appropriate chunk sizes (500-1500 tokens)
- Overlap chunks by 10-20% to prevent information loss
- Maintain document metadata for filtering and attribution

Retrieval Strategy
- Start with 5-10 retrieved documents
- Use hybrid search for better coverage
- Apply reranking to improve final results

Model Selection
- Choose embedding models based on your use case
- Balance model quality with computational resources
- Consider multilingual models for international applications

Evaluation
- Test with diverse queries
- Monitor retrieval quality and relevance
- Track answer accuracy and user satisfaction

Common Use Cases

1. Customer Support
RAG systems can provide accurate answers from product documentation and support tickets.

2. Research Assistance
Researchers can query large document collections for relevant information.

3. Internal Knowledge Management
Organizations can make internal documents searchable and accessible.

4. Educational Applications
Students can get answers from textbooks and course materials.

5. Legal and Compliance
Legal professionals can search through case law and regulations efficiently.

Conclusion

RAG represents a significant advancement in making AI systems more reliable, transparent, and useful. 
By combining retrieval with generation, these systems can provide accurate, sourced, and up-to-date 
information across a wide range of applications.
"""
    
    sample_file = documents_dir / "rag_introduction.txt"
    sample_file.write_text(sample_content)
    print(f"✓ Created sample document: {sample_file}")


def main():
    """Run setup"""
    print("="*60)
    print(" RAG System Setup")
    print("="*60)
    print()
    
    # Create directories
    print("Creating directories...")
    create_directories()
    print()
    
    # Create sample documents
    print("Creating sample documents...")
    create_sample_documents()
    print()
    
    print("="*60)
    print(" Setup Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. (Optional) Set up API keys in .env file")
    print("3. Run examples:")
    print("   - python examples/basic_usage.py")
    print("   - python examples/interactive_chat.py")
    print()


if __name__ == "__main__":
    main()

