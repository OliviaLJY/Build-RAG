#!/usr/bin/env python3
"""
Quick test to verify the query method is working
"""

from src.config import RAGConfig
from src.multimodal_rag import MultimodalRAGPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("\n" + "="*60)
print("Testing MultimodalRAGPipeline.query() method")
print("="*60)

try:
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    config = RAGConfig()
    pipeline = MultimodalRAGPipeline(config)
    print("   ✅ Pipeline initialized")
    
    # Try to load existing documents
    print("\n2. Loading existing documents...")
    try:
        pipeline.load_existing_store()
        print("   ✅ Documents loaded")
    except Exception as e:
        print(f"   ⚠️  No documents found: {e}")
        print("   Will use general knowledge")
    
    # Test the query method
    print("\n3. Testing query() method...")
    question = "What is machine learning?"
    print(f"   Question: {question}")
    
    result = pipeline.query(question, return_source_documents=True)
    
    print("\n   ✅ Query method works!")
    print(f"\n   Answer preview: {result['answer'][:200]}...")
    
    if 'source_documents' in result:
        print(f"   Sources: {len(result['source_documents'])} documents")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! The query() method is working correctly")
    print("="*60 + "\n")
    
except AttributeError as e:
    print(f"\n❌ ERROR: {e}")
    print("\nThe query() method is still missing.")
    print("="*60 + "\n")
    
except Exception as e:
    print(f"\n⚠️  ERROR: {e}")
    print("="*60 + "\n")

