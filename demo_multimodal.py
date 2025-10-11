"""
🎨 Interactive Demo: Multimodal RAG System

This script demonstrates all the multimodal capabilities of the enhanced RAG system.
Run this to see text, vision, and multimodal queries in action!
"""

import os
from pathlib import Path
from src.config import RAGConfig
from src.multimodal_rag import MultimodalRAGPipeline, MultimodalDocument
import logging
from PIL import Image, ImageDraw, ImageFont
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a nicely formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_subsection(title):
    """Print a subsection"""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


def create_demo_images():
    """Create sample images for demonstration"""
    print_section("🎨 Creating Demo Images")
    
    os.makedirs("./documents/demo", exist_ok=True)
    
    images = []
    
    # Image 1: Neural Network Diagram
    print("Creating neural network diagram...")
    img1 = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img1)
    
    # Draw neural network layers
    for layer in range(3):
        x = 150 + layer * 250
        for node in range(4):
            y = 150 + node * 100
            draw.ellipse([x-30, y-30, x+30, y+30], 
                        fill='lightblue', outline='darkblue', width=3)
    
    # Add title
    draw.text((300, 50), "Neural Network", fill='black')
    draw.text((250, 550), "Input → Hidden → Output", fill='darkblue')
    
    path1 = "./documents/demo/neural_network.jpg"
    img1.save(path1)
    images.append(("Neural Network Diagram", path1))
    print(f"✅ Saved: {path1}")
    
    # Image 2: Data Flow Chart
    print("Creating data flow chart...")
    img2 = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img2)
    
    # Draw flow chart
    boxes = [
        (100, 100, 300, 180, "Input Data"),
        (100, 250, 300, 330, "Processing"),
        (100, 400, 300, 480, "Output")
    ]
    
    for x1, y1, x2, y2, label in boxes:
        draw.rectangle([x1, y1, x2, y2], fill='lightgreen', outline='darkgreen', width=3)
        draw.text((x1+50, y1+30), label, fill='black')
        
    # Draw arrows
    draw.line([200, 180, 200, 250], fill='darkgreen', width=3)
    draw.line([200, 330, 200, 400], fill='darkgreen', width=3)
    
    draw.text((300, 50), "Data Pipeline", fill='black')
    
    path2 = "./documents/demo/data_flow.jpg"
    img2.save(path2)
    images.append(("Data Flow Chart", path2))
    print(f"✅ Saved: {path2}")
    
    # Image 3: Simple Chart
    print("Creating performance chart...")
    img3 = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img3)
    
    # Draw bar chart
    bars = [(100, 400), (250, 350), (400, 300), (550, 250)]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    
    for i, ((x, y), label) in enumerate(zip(bars, labels)):
        draw.rectangle([x, y, x+100, 500], fill='skyblue', outline='blue', width=2)
        draw.text((x+30, 520), label, fill='black')
    
    draw.text((300, 50), "Performance Metrics", fill='black')
    draw.text((250, 100), "Quarterly Results", fill='darkblue')
    
    path3 = "./documents/demo/chart.jpg"
    img3.save(path3)
    images.append(("Performance Chart", path3))
    print(f"✅ Saved: {path3}")
    
    print(f"\n✅ Created {len(images)} demo images")
    return images


def demo_1_text_query(pipeline):
    """Demonstrate standard text queries"""
    print_section("📝 DEMO 1: Text Query")
    
    questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the key concepts in deep learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print_subsection(f"Query {i}")
        print(f"❓ Question: {question}")
        
        start_time = time.time()
        result = pipeline.query(question, return_source_documents=True)
        elapsed = time.time() - start_time
        
        print(f"\n💡 Answer:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        print(f"\n⏱️  Response time: {elapsed:.2f}s")
        
        if result.get('source_documents'):
            print(f"📚 Sources: {len(result['source_documents'])} documents")
        
        time.sleep(1)  # Pause between queries
    
    print("\n✅ Text query demo completed!")


def demo_2_image_analysis(pipeline, demo_images):
    """Demonstrate image analysis"""
    print_section("🖼️  DEMO 2: Image Analysis")
    
    for name, path in demo_images[:2]:  # Analyze first 2 images
        print_subsection(f"Analyzing: {name}")
        print(f"📸 Image: {path}")
        
        try:
            start_time = time.time()
            result = pipeline.analyze_image(path)
            elapsed = time.time() - start_time
            
            if 'error' not in result:
                print(f"\n🔍 Analysis:")
                print(result['analysis'][:400] + "..." if len(result['analysis']) > 400 else result['analysis'])
                
                if result.get('embedding'):
                    print(f"\n📊 Embedding Info:")
                    print(f"   Dimension: {result['embedding_dimension']}")
                    print(f"   Sample values: {result['embedding'][:5]}")
                
                print(f"\n⏱️  Analysis time: {elapsed:.2f}s")
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            print(f"⚠️  Note: Ensure you have GPT-4 Vision access")
        
        time.sleep(1)
    
    print("\n✅ Image analysis demo completed!")


def demo_3_multimodal_query(pipeline, demo_images):
    """Demonstrate multimodal queries"""
    print_section("🎨 DEMO 3: Multimodal Query (Text + Image)")
    
    queries = [
        ("Neural Network Diagram", 
         "./documents/demo/neural_network.jpg",
         "Describe this neural network architecture and explain how it works"),
        
        ("Data Flow Chart",
         "./documents/demo/data_flow.jpg",
         "What process is shown in this diagram? How does it relate to data processing?"),
    ]
    
    for i, (name, path, question) in enumerate(queries, 1):
        print_subsection(f"Multimodal Query {i}: {name}")
        print(f"❓ Question: {question}")
        print(f"🖼️  Image: {path}")
        
        try:
            start_time = time.time()
            result = pipeline.query_multimodal(
                query_text=question,
                query_image=path,
                return_source_documents=True
            )
            elapsed = time.time() - start_time
            
            print(f"\n💡 Multimodal Answer:")
            print(result['answer'][:500] + "..." if len(result['answer']) > 500 else result['answer'])
            print(f"\n🎨 Multimodal: {result.get('multimodal', False)}")
            print(f"⏱️  Response time: {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Multimodal query failed: {e}")
            print(f"⚠️  Error occurred. Check your API key and GPT-4 Vision access.")
        
        time.sleep(1)
    
    print("\n✅ Multimodal query demo completed!")


def demo_4_comparison(pipeline, demo_images):
    """Compare different query modes"""
    print_section("📊 DEMO 4: Mode Comparison")
    
    image_path = "./documents/demo/neural_network.jpg"
    base_question = "What is a neural network?"
    
    print("Comparing three approaches to answer the same question:\n")
    
    # Mode 1: Text only
    print_subsection("Mode 1: Text Only")
    print(f"❓ Query: {base_question}")
    start = time.time()
    result1 = pipeline.query(base_question, return_source_documents=False)
    time1 = time.time() - start
    print(f"💡 Answer: {result1['answer'][:200]}...")
    print(f"⏱️  Time: {time1:.2f}s")
    
    time.sleep(1)
    
    # Mode 2: Image only
    print_subsection("Mode 2: Image Only")
    print(f"🖼️  Analyzing image without text query...")
    try:
        start = time.time()
        result2 = pipeline.analyze_image(image_path)
        time2 = time.time() - start
        if 'error' not in result2:
            print(f"💡 Analysis: {result2['analysis'][:200]}...")
            print(f"⏱️  Time: {time2:.2f}s")
        else:
            print(f"❌ Error: {result2['error']}")
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
    
    time.sleep(1)
    
    # Mode 3: Multimodal
    print_subsection("Mode 3: Multimodal (Text + Image)")
    print(f"❓ Query: {base_question}")
    print(f"🖼️  Image: {image_path}")
    try:
        start = time.time()
        result3 = pipeline.query_multimodal(
            query_text=base_question,
            query_image=image_path,
            return_source_documents=False
        )
        time3 = time.time() - start
        print(f"💡 Answer: {result3['answer'][:200]}...")
        print(f"⏱️  Time: {time3:.2f}s")
    except Exception as e:
        logger.error(f"Multimodal query failed: {e}")
    
    print("\n📈 Comparison Summary:")
    print(f"   Text Only:    Fast, relies on knowledge base or general knowledge")
    print(f"   Image Only:   Visual analysis, no text context")
    print(f"   Multimodal:   Best of both - visual + contextual understanding")
    
    print("\n✅ Comparison demo completed!")


def demo_5_system_stats(pipeline):
    """Show system statistics"""
    print_section("📊 DEMO 5: System Statistics")
    
    stats = pipeline.get_stats()
    
    print("System Configuration:")
    print(f"  • Total Documents:      {stats['num_documents']}")
    print(f"  • Multimodal Documents: {stats['num_multimodal_documents']}")
    print(f"  • Text-Only Documents:  {stats['num_text_only_documents']}")
    print(f"  • LLM Model:           {stats['llm_model']}")
    print(f"  • Embedding Model:     {stats['embedding_model']}")
    print(f"  • Vision Enabled:      {stats['vision_enabled']}")
    print(f"  • CLIP Available:      {stats['clip_available']}")
    
    print("\nCapabilities:")
    capabilities = [
        "✅ Text-based queries",
        "✅ Image analysis with GPT-4 Vision",
        "✅ Multimodal queries (text + image)",
        "✅ CLIP embeddings" if stats['clip_available'] else "⚠️  CLIP not available",
        "✅ Document retrieval",
        "✅ Semantic search",
        "✅ Visualization support"
    ]
    
    for cap in capabilities:
        print(f"  {cap}")
    
    print("\n✅ System stats demo completed!")


def demo_6_visualization_data(pipeline):
    """Show visualization capabilities"""
    print_section("📈 DEMO 6: Visualization Data")
    
    print("Getting embedding visualization data...")
    viz_data = pipeline.get_embedding_visualization_data()
    
    print(f"\nVisualization Info:")
    print(f"  • Total documents:      {viz_data.get('num_documents', 0)}")
    print(f"  • Labels available:     {len(viz_data.get('labels', []))}")
    print(f"  • Multimodal enabled:   {viz_data.get('multimodal_enabled', False)}")
    
    if viz_data.get('labels'):
        print(f"\nSample document labels:")
        for i, label in enumerate(viz_data['labels'][:3], 1):
            print(f"  {i}. {label}")
    
    print("\n💡 Visualization Features:")
    features = [
        "📊 Embedding vector plots (Plotly)",
        "📈 Response metrics charts",
        "🎯 Real-time system monitoring",
        "🎨 Interactive dashboards",
        "📉 Performance analytics"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n✅ Visualization demo completed!")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("  🎨 MULTIMODAL RAG SYSTEM - INTERACTIVE DEMO")
    print("="*70)
    print("\nThis demo will showcase all multimodal capabilities:")
    print("  1. Text queries")
    print("  2. Image analysis")
    print("  3. Multimodal queries (text + image)")
    print("  4. Mode comparisons")
    print("  5. System statistics")
    print("  6. Visualization features")
    
    input("\n📌 Press Enter to start the demo...")
    
    # Initialize pipeline
    print_section("🚀 Initializing Multimodal RAG Pipeline")
    print("Loading models and components...")
    
    try:
        config = RAGConfig()
        pipeline = MultimodalRAGPipeline(config)
        print("✅ Pipeline initialized successfully!")
        
        # Try to load existing documents
        try:
            pipeline.vector_store_manager.load_vector_store()
            pipeline._initialize_retriever()
            print("✅ Loaded existing document store")
        except Exception as e:
            logger.warning(f"No existing document store: {e}")
            print("⚠️  No documents loaded - will use general knowledge")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        print(f"\n❌ Initialization failed: {e}")
        print("\nPlease check:")
        print("  1. OpenAI API key is set in .env")
        print("  2. Required packages are installed")
        print("  3. Internet connection is available")
        return
    
    # Create demo images
    try:
        demo_images = create_demo_images()
    except Exception as e:
        logger.error(f"Failed to create demo images: {e}")
        demo_images = []
    
    # Run demos
    try:
        # Demo 1: Text queries
        input("\n📌 Press Enter to run Demo 1: Text Queries...")
        demo_1_text_query(pipeline)
        
        # Demo 2: Image analysis
        if demo_images:
            input("\n📌 Press Enter to run Demo 2: Image Analysis...")
            demo_2_image_analysis(pipeline, demo_images)
        
        # Demo 3: Multimodal queries
        if demo_images:
            input("\n📌 Press Enter to run Demo 3: Multimodal Queries...")
            demo_3_multimodal_query(pipeline, demo_images)
        
        # Demo 4: Comparison
        if demo_images:
            input("\n📌 Press Enter to run Demo 4: Mode Comparison...")
            demo_4_comparison(pipeline, demo_images)
        
        # Demo 5: System stats
        input("\n📌 Press Enter to run Demo 5: System Statistics...")
        demo_5_system_stats(pipeline)
        
        # Demo 6: Visualization
        input("\n📌 Press Enter to run Demo 6: Visualization Features...")
        demo_6_visualization_data(pipeline)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\n❌ Error during demo: {e}")
    
    # Conclusion
    print_section("🎉 Demo Complete!")
    
    print("You've seen the multimodal RAG system in action!")
    print("\nKey Takeaways:")
    print("  ✨ Handles text, images, and multimodal queries")
    print("  🚀 Powered by GPT-4o Vision and CLIP")
    print("  📊 Rich visualizations and analytics")
    print("  ⚡ Fast, scalable, and production-ready")
    
    print("\n🚀 Next Steps:")
    print("  1. Start the API server:     python api_server_multimodal.py")
    print("  2. Open the frontend:        frontend_multimodal.html")
    print("  3. Read the guide:          MULTIMODAL_GUIDE.md")
    print("  4. Ingest your documents:   Use the API or Python SDK")
    print("  5. Start building!          Create amazing AI applications")
    
    print("\n" + "="*70)
    print("  Happy building with Multimodal RAG! 🎨🚀")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

