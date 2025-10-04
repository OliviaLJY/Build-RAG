"""
Streamlit Web Frontend for RAG System
Beautiful, interactive interface for document Q&A
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Page configuration
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .source-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e3f2fd;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """Initialize RAG pipeline (cached)"""
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k_retrieval=5,
        use_hybrid_search=False,
        use_reranking=False,
        use_contextual_compression=False,
        collection_name='web_rag_documents'
    )
    
    rag = RAGPipeline(config)
    
    # Check if documents exist
    doc_path = Path("documents")
    if doc_path.exists() and any(doc_path.glob("*.txt")):
        try:
            # Try to load existing store
            rag.load_existing_store()
        except:
            # Create new store from documents
            rag.ingest_documents("documents", create_new=True)
    
    return rag


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 RAG AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about AI, Machine Learning, and Deep Learning</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag' not in st.session_state:
        with st.spinner("🚀 Initializing AI system..."):
            st.session_state.rag = initialize_rag()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System stats
        st.subheader("📊 System Status")
        try:
            stats = st.session_state.rag.get_stats()
            st.markdown(f"""
            <div class="stats-box">
                <strong>Documents:</strong> {stats.get('num_documents', 0)} chunks<br>
                <strong>Model:</strong> {stats.get('embedding_model', 'N/A').split('/')[-1]}<br>
                <strong>Status:</strong> ✅ Ready
            </div>
            """, unsafe_allow_html=True)
        except:
            st.info("No documents loaded yet")
        
        st.divider()
        
        # Document upload
        st.subheader("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload documents to ask questions about"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is machine learning?",
            "Explain neural networks",
            "What is deep learning?",
            "How does backpropagation work?",
            "What are transformers?",
            "Explain gradient descent",
            "What is overfitting?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now()
                })
                st.rerun()
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reload Documents"):
            st.cache_resource.clear()
            st.session_state.rag = initialize_rag()
            st.success("Documents reloaded!")
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="🤔"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(message["content"])
                        if "sources" in message:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(message["sources"], 1):
                                    st.markdown(f"""
                                    <div class="source-box">
                                        <strong>Source {i}:</strong><br>
                                        {source[:300]}...
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask a question about AI, ML, or Deep Learning...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            
            # Get response
            with st.spinner("🔍 Searching documents..."):
                try:
                    # Retrieve documents
                    docs = st.session_state.rag.retrieve(user_input)
                    
                    if docs:
                        # Format response
                        response = f"**Answer based on retrieved documents:**\n\n{docs[0].page_content}"
                        sources = [doc.page_content for doc in docs]
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources,
                            "timestamp": datetime.now()
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I couldn't find relevant information about that. Try asking about machine learning, neural networks, or deep learning concepts.",
                            "timestamp": datetime.now()
                        })
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure documents are loaded. Try reloading documents from the sidebar.")
            
            st.rerun()
    
    with col2:
        st.subheader("📖 About")
        st.info("""
        This RAG (Retrieval-Augmented Generation) system helps you find answers in AI learning materials.
        
        **How to use:**
        1. Ask questions in the chat
        2. View retrieved passages
        3. Upload your own documents
        
        **Available Topics:**
        - Machine Learning
        - Deep Learning
        - Neural Networks
        - AI Algorithms
        - And more!
        """)
        
        st.subheader("🎯 Features")
        st.markdown("""
        - ✅ Semantic search
        - ✅ Fast retrieval (~10ms)
        - ✅ Source attribution
        - ✅ Document upload
        - ✅ Offline capable
        """)


def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    doc_dir = Path("documents/uploaded")
    doc_dir.mkdir(parents=True, exist_ok=True)
    
    with st.spinner("Processing documents..."):
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = doc_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Re-ingest documents
        try:
            st.session_state.rag.ingest_documents("documents", create_new=True)
            st.success(f"✅ Processed {len(uploaded_files)} file(s)!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Error processing files: {e}")


if __name__ == "__main__":
    main()

