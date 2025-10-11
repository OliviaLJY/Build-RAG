"""
Test script for Multimodal RAG System
Demonstrates text, image, and combined queries
"""

import os
from pathlib import Path
from src.config import RAGConfig
from src.multimodal_rag import MultimodalRAGPipeline, MultimodalDocument
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_query():
    """Test standard text query"""
    print("\n" + "="*60)
    print("TEST 1: Standard Text Query")
    print("="*60)
    
    config = RAGConfig()
    pipeline = MultimodalRAGPipeline(config)
    
    # Load existing documents
    try:
        pipeline.vector_store_manager.load_vector_store()
        pipeline._initialize_retriever()
        logger.info("✅ Loaded existing vector store")
    except Exception as e:
        logger.error(f"❌ No vector store found. Please ingest documents first: {e}")
        return
    
    # Test query
    question = "What is machine learning?"
    print(f"\n📝 Question: {question}")
    
    result = pipeline.query(question, return_source_documents=True)
    
    print(f"\n💡 Answer:\n{result['answer']}")
    
    if 'source_documents' in result and result['source_documents']:
        print(f"\n📚 Sources ({len(result['source_documents'])}):")
        for i, doc in enumerate(result['source_documents'][:3], 1):
            print(f"  {i}. {doc.metadata.get('source', 'Unknown')}")
    
    print("\n✅ Text query test completed!")


def test_image_analysis():
    """Test image analysis with GPT-4 Vision"""
    print("\n" + "="*60)
    print("TEST 2: Image Analysis (GPT-4 Vision)")
    print("="*60)
    
    # For this test, you need to have an image file
    # Let's create a simple test case
    
    config = RAGConfig()
    pipeline = MultimodalRAGPipeline(config)
    
    # Check if there's a test image
    test_image_path = "./documents/test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"\n⚠️  No test image found at {test_image_path}")
        print("To test image analysis:")
        print("1. Place an image at ./documents/test_image.jpg")
        print("2. Or use the API endpoint with image upload")
        return
    
    print(f"\n🖼️  Analyzing image: {test_image_path}")
    
    try:
        result = pipeline.analyze_image(test_image_path)
        
        print(f"\n🔍 Analysis:\n{result['analysis']}")
        
        if result.get('embedding'):
            print(f"\n📊 Embedding dimension: {result['embedding_dimension']}")
            print(f"First 10 values: {result['embedding'][:10]}")
        
        print("\n✅ Image analysis test completed!")
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {e}")
        print("Note: Make sure you have a valid OpenAI API key with GPT-4 Vision access")


def test_multimodal_query():
    """Test multimodal query (text + image)"""
    print("\n" + "="*60)
    print("TEST 3: Multimodal Query (Text + Image)")
    print("="*60)
    
    config = RAGConfig()
    pipeline = MultimodalRAGPipeline(config)
    
    # Load existing documents
    try:
        pipeline.vector_store_manager.load_vector_store()
        pipeline._initialize_retriever()
    except Exception as e:
        logger.warning(f"No vector store found: {e}")
    
    test_image_path = "./documents/test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"\n⚠️  No test image found at {test_image_path}")
        return
    
    question = "What do you see in this image? How does it relate to machine learning?"
    print(f"\n💬 Question: {question}")
    print(f"🖼️  Image: {test_image_path}")
    
    try:
        result = pipeline.query_multimodal(
            query_text=question,
            query_image=test_image_path,
            return_source_documents=True
        )
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n🎨 Multimodal: {result['multimodal']}")
        
        print("\n✅ Multimodal query test completed!")
        
    except Exception as e:
        logger.error(f"❌ Multimodal query failed: {e}")


def test_system_stats():
    """Test system statistics"""
    print("\n" + "="*60)
    print("TEST 4: System Statistics")
    print("="*60)
    
    config = RAGConfig()
    pipeline = MultimodalRAGPipeline(config)
    
    # Load existing documents if available
    try:
        pipeline.vector_store_manager.load_vector_store()
        pipeline._initialize_retriever()
    except:
        pass
    
    stats = pipeline.get_stats()
    
    print("\n📊 System Statistics:")
    print(f"  • Total documents: {stats['num_documents']}")
    print(f"  • Multimodal documents: {stats['num_multimodal_documents']}")
    print(f"  • Text-only documents: {stats['num_text_only_documents']}")
    print(f"  • CLIP embeddings available: {stats['multimodal_embeddings_available']}")
    print(f"  • Vision enabled: {stats['vision_enabled']}")
    print(f"  • LLM model: {stats['llm_model']}")
    print(f"  • Embedding model: {stats['embedding_model']}")
    
    print("\n✅ Statistics test completed!")


def create_test_image_placeholder():
    """Create a simple test image if PIL is available"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes
        draw.rectangle([100, 100, 700, 500], outline='blue', width=5)
        draw.ellipse([200, 200, 600, 400], fill='lightblue', outline='darkblue', width=3)
        
        # Add text
        draw.text((250, 280), "Machine Learning", fill='black')
        draw.text((300, 320), "Test Image", fill='darkblue')
        
        # Save
        output_path = "./documents/test_image.jpg"
        os.makedirs("./documents", exist_ok=True)
        img.save(output_path)
        
        print(f"✅ Created test image at {output_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not create test image: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🎨 MULTIMODAL RAG SYSTEM TESTS")
    print("="*60)
    
    # Check for test image
    if not os.path.exists("./documents/test_image.jpg"):
        print("\n📸 Creating test image...")
        if not create_test_image_placeholder():
            print("⚠️  Could not create test image. Image tests will be skipped.")
    
    # Run tests
    try:
        test_text_query()
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
    
    try:
        test_image_analysis()
    except Exception as e:
        logger.error(f"Test 2 failed: {e}")
    
    try:
        test_multimodal_query()
    except Exception as e:
        logger.error(f"Test 3 failed: {e}")
    
    try:
        test_system_stats()
    except Exception as e:
        logger.error(f"Test 4 failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the API server: python api_server_multimodal.py")
    print("2. Open frontend_multimodal.html in your browser")
    print("3. Create an API key: curl -X POST http://localhost:8000/api/keys \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"name\":\"test-key\",\"user_id\":\"test-user\"}'")
    print("4. Use the API key in the frontend to start querying!")
    print("\n")


if __name__ == "__main__":
    main()

