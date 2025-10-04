"""
Configuration management for RAG system
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Configuration for RAG system"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    cohere_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))
    
    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Model for generating embeddings"
    )
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="Language model for generation"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
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
    max_tokens: int = Field(default=2000)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = RAGConfig()

