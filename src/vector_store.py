"""
Vector store management with ChromaDB and FAISS support
"""

from typing import List, Optional
import logging
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain.embeddings.base import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manager for vector store operations with multiple backends
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        store_type: str = "chromadb",
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/vectorstore"
    ):
        """
        Initialize vector store manager
        
        Args:
            embeddings: Embeddings instance
            store_type: Type of vector store ('chromadb' or 'faiss')
            collection_name: Name of the collection
            persist_directory: Directory to persist the vector store
        """
        self.embeddings = embeddings
        self.store_type = store_type.lower()
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store = None
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """
        Create and populate vector store from documents
        
        Args:
            documents: List of documents to add to vector store
        """
        try:
            if self.store_type == "chromadb":
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                logger.info(f"Created ChromaDB vector store with {len(documents)} documents")
                
            elif self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                # Persist FAISS index
                self.vector_store.save_local(self.persist_directory)
                logger.info(f"Created FAISS vector store with {len(documents)} documents")
                
            else:
                raise ValueError(f"Unsupported vector store type: {self.store_type}")
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self) -> None:
        """Load existing vector store from disk"""
        try:
            if self.store_type == "chromadb":
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                logger.info("Loaded ChromaDB vector store")
                
            elif self.store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings
                )
                logger.info("Loaded FAISS vector store")
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vector store
        
        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        try:
            self.vector_store.add_documents(documents)
            
            # Persist changes
            if self.store_type == "chromadb":
                self.vector_store.persist()
            elif self.store_type == "faiss":
                self.vector_store.save_local(self.persist_directory)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query text
            k: Number of documents to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {e}")
            raise
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get retriever interface for the vector store
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        if self.store_type == "chromadb" and self.vector_store:
            self.vector_store.delete_collection()
            logger.info(f"Deleted collection: {self.collection_name}")

