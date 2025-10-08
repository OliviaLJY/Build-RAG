"""
Complete RAG pipeline integrating all components
"""

from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from .config import RAGConfig
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retrieval import AdvancedRetriever, CohereReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for document ingestion, retrieval, and generation
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline
        
        Args:
            config: RAG configuration object
        """
        self.config = config or RAGConfig()
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy="recursive"
        )
        
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embedding_manager.get_embeddings_instance(),
            store_type=self.config.vector_store_type,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory
        )
        
        self.documents = []
        self.retriever = None
        self.reranker = None
        
        # Initialize reranker if enabled
        if self.config.use_reranking and self.config.cohere_api_key:
            try:
                self.reranker = CohereReranker(self.config.cohere_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
        
        logger.info("RAG Pipeline initialized")
    
    def ingest_documents(
        self,
        source_path: str,
        create_new: bool = True
    ) -> None:
        """
        Ingest documents from a file or directory
        
        Args:
            source_path: Path to file or directory
            create_new: Whether to create new vector store or add to existing
        """
        try:
            # Load documents
            logger.info(f"Loading documents from: {source_path}")
            raw_documents = self.doc_processor.load_documents(source_path)
            
            # Process and chunk documents
            logger.info("Processing and chunking documents...")
            self.documents = self.doc_processor.process_documents(raw_documents)
            
            # Create or update vector store
            if create_new:
                logger.info("Creating vector store...")
                self.vector_store_manager.create_vector_store(self.documents)
            else:
                logger.info("Adding to existing vector store...")
                self.vector_store_manager.load_vector_store()
                self.vector_store_manager.add_documents(self.documents)
            
            # Initialize retriever
            self._initialize_retriever()
            
            logger.info(f"Successfully ingested {len(self.documents)} document chunks")
            
        except Exception as e:
            logger.error(f"Error ingesting documents: {e}")
            raise
    
    def load_existing_store(self) -> None:
        """Load existing vector store from disk"""
        try:
            logger.info("Loading existing vector store...")
            self.vector_store_manager.load_vector_store()
            self._initialize_retriever()
            logger.info("Vector store loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def _initialize_retriever(self) -> None:
        """Initialize the retriever with advanced features"""
        base_retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.config.top_k_retrieval}
        )
        
        self.retriever = AdvancedRetriever(
            vector_retriever=base_retriever,
            documents=self.documents,
            use_hybrid_search=self.config.use_hybrid_search,
            use_compression=self.config.use_contextual_compression,
            top_k=self.config.top_k_retrieval,
            openai_api_key=self.config.openai_api_key
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Ingest or load documents first.")
        
        # Retrieve documents
        documents = self.retriever.retrieve(query, top_k=self.config.top_k_retrieval)
        
        # Apply reranking if enabled
        if self.reranker and len(documents) > 0:
            documents = self.reranker.rerank(
                query,
                documents,
                top_k=self.config.rerank_top_k
            )
        
        return documents
    
    def query(
        self,
        question: str,
        return_source_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with 'answer' and optionally 'source_documents'
        """
        try:
            # Retrieve relevant documents
            documents = self.retrieve(question)
            
            # Initialize LLM (needed for both document-based and general knowledge answers)
            llm = ChatOpenAI(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                openai_api_key=self.config.openai_api_key,
                max_tokens=self.config.max_tokens
            )
            
            # Handle case when no documents are found
            if not documents:
                if self.config.allow_general_knowledge:
                    # Answer from GPT-4o's general knowledge with disclaimer
                    logger.info("No documents found, using GPT-4o's general knowledge")
                    
                    general_prompt = f"""You are a knowledgeable AI assistant. Answer the following question accurately and helpfully based on your training knowledge.

Note: No relevant documents were found in the knowledge base, so please answer from your general knowledge.

Question: {question}

Please provide a comprehensive and accurate answer.

Answer:"""
                    
                    try:
                        answer = llm.predict(general_prompt)
                        return {
                            "answer": f"ℹ️ **Note:** No relevant documents found in knowledge base. Answer based on general AI knowledge:\n\n{answer}",
                            "source_documents": [],
                            "knowledge_source": "general_knowledge"
                        }
                    except Exception as e:
                        logger.error(f"Error generating general knowledge answer: {e}")
                        return {
                            "answer": "I couldn't find relevant information in the knowledge base and encountered an error accessing general knowledge.",
                            "source_documents": []
                        }
                else:
                    # Strict mode: only answer from documents
                    logger.info("No documents found, strict mode enabled")
                    return {
                        "answer": "I couldn't find any relevant information to answer your question in the knowledge base.",
                        "source_documents": [],
                        "knowledge_source": "none"
                    }
            
            # Create enhanced intelligent prompt
            prompt_template = """You are an intelligent AI assistant with access to a knowledge base. Your task is to provide accurate, insightful, and well-reasoned answers based on the context provided.

Context Information:
{context}

Question: {question}

Instructions:
1. Analyze the context carefully and extract the most relevant information
2. Provide a comprehensive yet concise answer that directly addresses the question
3. Use clear reasoning and connect related concepts when appropriate
4. If the context doesn't contain enough information, acknowledge this and provide what you can based on available information
5. Be precise with technical terms and explanations
6. When relevant, provide examples or clarifications to enhance understanding

Intelligent Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever.retriever,
                return_source_documents=return_source_documents,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Get answer
            result = qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
            }
            
            if return_source_documents:
                response["source_documents"] = result["source_documents"]
            
            logger.info(f"Generated answer for query: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface
        
        Args:
            question: Question to ask
            
        Returns:
            Answer string
        """
        result = self.query(question, return_source_documents=False)
        return result["answer"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "num_documents": len(self.documents),
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "vector_store_type": self.config.vector_store_type,
            "hybrid_search_enabled": self.config.use_hybrid_search,
            "reranking_enabled": self.config.use_reranking,
        }

