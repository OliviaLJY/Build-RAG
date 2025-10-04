"""
Automated test script for RAG system
Run this to verify all components are working correctly
"""

import sys
import time
from pathlib import Path

# Color output for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def test_imports():
    """Test 1: Verify all required packages are installed"""
    print_header("Test 1: Checking Dependencies")
    
    required_packages = {
        'langchain': 'LangChain',
        'chromadb': 'ChromaDB',
        'sentence_transformers': 'Sentence Transformers',
        'faiss': 'FAISS (CPU)',
        'pypdf': 'PyPDF',
        'tiktoken': 'Tiktoken',
        'numpy': 'NumPy',
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} NOT installed")
            all_installed = False
    
    return all_installed


def test_document_processing():
    """Test 2: Document loading and chunking"""
    print_header("Test 2: Document Processing")
    
    try:
        from src.document_processor import DocumentProcessor
        
        # Check if sample document exists
        doc_path = Path("documents/rag_introduction.txt")
        if not doc_path.exists():
            print_error("Sample document not found. Run setup.py first.")
            return False
        
        # Test loading
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        docs = processor.load_documents(str(doc_path))
        print_success(f"Loaded {len(docs)} document(s)")
        
        # Test chunking
        chunks = processor.process_documents(docs)
        print_success(f"Created {len(chunks)} chunks")
        
        if len(chunks) > 0:
            print_info(f"Chunk size: ~{len(chunks[0].page_content)} characters")
            print_info(f"First chunk preview: {chunks[0].page_content[:80]}...")
        
        return True
        
    except Exception as e:
        print_error(f"Document processing failed: {e}")
        return False


def test_embeddings():
    """Test 3: Embedding generation"""
    print_header("Test 3: Embedding Generation")
    
    try:
        from src.embeddings import EmbeddingManager
        
        # Use smaller model for faster testing
        print_info("Initializing embedding model (this may take a moment)...")
        manager = EmbeddingManager(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        print_success("Embedding manager initialized")
        
        # Test query embedding
        start = time.time()
        embedding = manager.embed_query("What is RAG?")
        duration = time.time() - start
        
        print_success(f"Generated embedding with {len(embedding)} dimensions")
        print_info(f"Embedding generation took {duration:.3f}s")
        
        # Test batch embedding
        texts = ["First text", "Second text", "Third text"]
        embeddings = manager.embed_documents(texts)
        print_success(f"Batch embedded {len(embeddings)} documents")
        
        return True
        
    except Exception as e:
        print_error(f"Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test 4: Vector store operations"""
    print_header("Test 4: Vector Store")
    
    try:
        from src.embeddings import EmbeddingManager
        from src.vector_store import VectorStoreManager
        from langchain.docstore.document import Document
        
        # Initialize
        embeddings = EmbeddingManager(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        vector_store = VectorStoreManager(
            embeddings=embeddings.get_embeddings_instance(),
            store_type='chromadb',
            collection_name='test_collection',
            persist_directory='./data/test_vectorstore'
        )
        print_success("Vector store manager initialized")
        
        # Create test documents
        test_docs = [
            Document(
                page_content="Python is a high-level programming language.",
                metadata={"topic": "programming"}
            ),
            Document(
                page_content="Machine learning is a subset of AI.",
                metadata={"topic": "ai"}
            ),
            Document(
                page_content="RAG combines retrieval with generation.",
                metadata={"topic": "rag"}
            ),
        ]
        
        # Store documents
        vector_store.create_vector_store(test_docs)
        print_success(f"Created vector store with {len(test_docs)} documents")
        
        # Test search
        start = time.time()
        results = vector_store.similarity_search("What is RAG?", k=2)
        search_time = time.time() - start
        
        print_success(f"Search completed in {search_time*1000:.2f}ms")
        print_success(f"Retrieved {len(results)} results")
        
        if len(results) > 0:
            print_info(f"Top result: {results[0].page_content[:50]}...")
        
        # Test with scores
        results_with_scores = vector_store.similarity_search_with_score("RAG", k=1)
        if len(results_with_scores) > 0:
            doc, score = results_with_scores[0]
            print_info(f"Top result relevance score: {score:.4f}")
        
        return True
        
    except Exception as e:
        print_error(f"Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval():
    """Test 5: Advanced retrieval"""
    print_header("Test 5: Advanced Retrieval")
    
    try:
        from src.embeddings import EmbeddingManager
        from src.vector_store import VectorStoreManager
        from src.retrieval import AdvancedRetriever
        from langchain.docstore.document import Document
        
        # Setup
        embeddings = EmbeddingManager(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        vector_store = VectorStoreManager(
            embeddings=embeddings.get_embeddings_instance(),
            store_type='chromadb',
            collection_name='test_retrieval'
        )
        
        # Create documents
        docs = [
            Document(page_content=f"Document {i} about machine learning and AI.") 
            for i in range(10)
        ]
        docs.append(Document(page_content="RAG is a powerful technique for AI applications."))
        
        vector_store.create_vector_store(docs)
        print_success("Created test vector store")
        
        # Test retrieval
        base_retriever = vector_store.get_retriever(search_kwargs={"k": 5})
        
        retriever = AdvancedRetriever(
            vector_retriever=base_retriever,
            documents=docs,
            use_hybrid_search=True,
            use_compression=False,  # Skip to avoid OpenAI requirement
            top_k=3
        )
        print_success("Initialized advanced retriever")
        
        results = retriever.retrieve("What is RAG?")
        print_success(f"Retrieved {len(results)} documents")
        
        return True
        
    except Exception as e:
        print_error(f"Retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test 6: Full RAG pipeline (without LLM)"""
    print_header("Test 6: Full RAG Pipeline")
    
    try:
        from src.rag_pipeline import RAGPipeline
        from src.config import RAGConfig
        
        # Configure without LLM requirements
        config = RAGConfig(
            chunk_size=500,
            chunk_overlap=100,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            top_k_retrieval=5,
            use_hybrid_search=False,  # Simplify for testing
            use_reranking=False,
            use_contextual_compression=False,
            collection_name='test_pipeline'
        )
        print_success("Configuration created")
        
        # Initialize pipeline
        rag = RAGPipeline(config)
        print_success("RAG pipeline initialized")
        
        # Ingest documents
        doc_path = "documents/rag_introduction.txt"
        if not Path(doc_path).exists():
            print_error("Sample document not found")
            return False
        
        start = time.time()
        rag.ingest_documents(doc_path, create_new=True)
        ingest_time = time.time() - start
        print_success(f"Documents ingested in {ingest_time:.2f}s")
        
        # Test retrieval
        query = "What is RAG?"
        start = time.time()
        docs = rag.retrieve(query)
        retrieve_time = time.time() - start
        
        print_success(f"Retrieved {len(docs)} documents in {retrieve_time:.3f}s")
        
        if len(docs) > 0:
            print_info(f"Top result: {docs[0].page_content[:100]}...")
        
        # Show stats
        stats = rag.get_stats()
        print_info(f"Total chunks: {stats['num_documents']}")
        
        return True
        
    except Exception as e:
        print_error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test 7: Performance benchmarks"""
    print_header("Test 7: Performance Benchmarks")
    
    try:
        from src.document_processor import DocumentProcessor
        from src.embeddings import EmbeddingManager
        import time
        
        # Test document processing speed
        large_text = "This is a test sentence about artificial intelligence. " * 500
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        start = time.time()
        chunks = processor.process_text(large_text)
        proc_time = time.time() - start
        
        print_success(f"Processed {len(large_text):,} characters in {proc_time:.3f}s")
        print_info(f"Speed: {len(large_text)/proc_time:,.0f} chars/sec")
        print_info(f"Created {len(chunks)} chunks")
        
        # Test embedding speed
        manager = EmbeddingManager(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        test_texts = ["Test query " + str(i) for i in range(10)]
        start = time.time()
        embeddings = manager.embed_documents(test_texts)
        embed_time = time.time() - start
        
        print_success(f"Generated {len(embeddings)} embeddings in {embed_time:.3f}s")
        print_info(f"Speed: {len(embeddings)/embed_time:.1f} embeddings/sec")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False


def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}RAG System - Automated Test Suite{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    # Check setup
    if not Path("documents").exists():
        print_error("Documents directory not found!")
        print_info("Please run: python setup.py")
        return
    
    if not Path("documents/rag_introduction.txt").exists():
        print_error("Sample document not found!")
        print_info("Please run: python setup.py")
        return
    
    # Run tests
    tests = [
        ("Dependencies", test_imports),
        ("Document Processing", test_document_processing),
        ("Embeddings", test_embeddings),
        ("Vector Store", test_vector_store),
        ("Retrieval", test_retrieval),
        ("Full Pipeline", test_full_pipeline),
        ("Performance", test_performance),
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    total_time = time.time() - start_time
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}")
        else:
            print_error(f"{name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    print(f"{Colors.BOLD}Total time: {total_time:.2f}s{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Your RAG system is ready to use.{Colors.END}\n")
        print("Next steps:")
        print("  1. Try: python examples/interactive_chat.py")
        print("  2. Add your documents to the 'documents/' folder")
        print("  3. Read QUICKSTART.md for usage examples\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed. Check the errors above.{Colors.END}\n")
        print("Troubleshooting:")
        print("  1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Run setup: python setup.py")
        print("  3. Check TEST_GUIDE.md for detailed help\n")


if __name__ == "__main__":
    main()

