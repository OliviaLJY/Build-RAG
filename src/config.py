"""
Configuration management for RAG system
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RAGConfig(BaseModel):
    """Configuration for RAG system"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    cohere_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    
    # Model Configuration - Enhanced with OpenAI models for better intelligence
    embedding_model: str = Field(
        default="text-embedding-3-large",  # Upgraded to OpenAI embeddings for better semantic understanding
        description="Model for generating embeddings"
    )
    llm_model: str = Field(
        default="gpt-4o",  # Upgraded to GPT-4o - latest and most intelligent model
        description="Language model for generation"
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)  # Lower temperature for more focused, accurate responses
    
    # Vector Store Configuration
    vector_store_type: str = Field(default="chromadb", description="Type of vector store")
    collection_name: str = Field(default="rag_documents")
    persist_directory: str = Field(default="./data/vectorstore")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(default=5, description="Number of documents to retrieve")
    rerank_top_k: int = Field(default=3, description="Number of documents after reranking")
    use_reranking: bool = Field(default=True, description="Enable reranking")
    use_hybrid_search: bool = Field(default=True, description="Enable hybrid search")
    
    # Advanced Options
    use_contextual_compression: bool = Field(default=True)
    max_tokens: int = Field(default=4000)  # Increased for more detailed, intelligent responses
    allow_general_knowledge: bool = Field(
        default=True,  # Allow GPT-4o to answer even without documents
        description="Allow answering from GPT-4o's training when no documents found"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = RAGConfig()

