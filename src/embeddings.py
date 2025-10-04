"""
Embedding generation with multiple model support
"""

from typing import List, Optional
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manager for generating embeddings with multiple models
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the embedding model
            openai_api_key: OpenAI API key if using OpenAI embeddings
        """
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.embeddings = self._get_embeddings()
    
    def _get_embeddings(self):
        """Initialize embeddings based on model type"""
        try:
            if self.model_name.startswith("text-embedding"):
                # OpenAI embeddings
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key required for OpenAI embeddings")
                
                logger.info(f"Initializing OpenAI embeddings: {self.model_name}")
                return OpenAIEmbeddings(
                    model=self.model_name,
                    openai_api_key=self.openai_api_key
                )
            else:
                # HuggingFace embeddings (default)
                logger.info(f"Initializing HuggingFace embeddings: {self.model_name}")
                return HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def get_embeddings_instance(self):
        """Get the underlying embeddings instance for use with vector stores"""
        return self.embeddings


# Recommended embedding models for different use cases
EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-mpnet-base-v2",  # Best all-around
    "fast": "sentence-transformers/all-MiniLM-L6-v2",      # Fast and efficient
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "openai": "text-embedding-3-small",                     # OpenAI's latest
    "large": "sentence-transformers/all-mpnet-base-v2",    # High quality
}

