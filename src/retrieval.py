"""
Advanced retrieval engine with hybrid search and reranking
"""

from typing import List, Optional
import logging

from langchain.docstore.document import Document
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    BM25Retriever
)
from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter
)
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Advanced retrieval with hybrid search, reranking, and compression
    """
    
    def __init__(
        self,
        vector_retriever,
        documents: Optional[List[Document]] = None,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_compression: bool = True,
        top_k: int = 5,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize advanced retriever
        
        Args:
            vector_retriever: Base vector store retriever
            documents: Optional list of documents for BM25
            use_hybrid_search: Enable hybrid search (semantic + keyword)
            use_reranking: Enable reranking
            use_compression: Enable contextual compression
            top_k: Number of documents to retrieve
            openai_api_key: OpenAI API key for LLM-based features
        """
        self.vector_retriever = vector_retriever
        self.documents = documents or []
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_compression = use_compression
        self.top_k = top_k
        self.openai_api_key = openai_api_key
        
        self.retriever = self._build_retriever()
    
    def _build_retriever(self):
        """Build the retrieval pipeline"""
        base_retriever = self.vector_retriever
        
        # Add hybrid search (combine semantic and keyword-based)
        if self.use_hybrid_search and self.documents:
            base_retriever = self._add_hybrid_search(base_retriever)
        
        # Add contextual compression
        if self.use_compression:
            base_retriever = self._add_compression(base_retriever)
        
        return base_retriever
    
    def _add_hybrid_search(self, vector_retriever):
        """
        Add hybrid search combining semantic and keyword-based retrieval
        
        BM25 (Best Matching 25) is a keyword-based algorithm that excels at
        finding exact matches, while vector search is better for semantic similarity.
        """
        try:
            # Create BM25 retriever for keyword-based search
            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = self.top_k
            
            # Ensemble retriever combines both approaches
            # Weights: 0.5 for semantic, 0.5 for keyword (can be adjusted)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            
            logger.info("Enabled hybrid search (semantic + BM25)")
            return ensemble_retriever
            
        except Exception as e:
            logger.warning(f"Could not enable hybrid search: {e}")
            return vector_retriever
    
    def _add_compression(self, base_retriever):
        """
        Add contextual compression to filter and compress retrieved documents
        
        This removes irrelevant parts and keeps only the most relevant context.
        """
        try:
            if self.openai_api_key:
                # LLM-based compression (more accurate but slower)
                llm = ChatOpenAI(
                    temperature=0,
                    model_name="gpt-3.5-turbo",
                    openai_api_key=self.openai_api_key
                )
                compressor = LLMChainExtractor.from_llm(llm)
                logger.info("Enabled LLM-based contextual compression")
            else:
                # Embeddings-based compression (faster)
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings()
                compressor = EmbeddingsFilter(
                    embeddings=embeddings,
                    similarity_threshold=0.76
                )
                logger.info("Enabled embeddings-based compression")
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            return compression_retriever
            
        except Exception as e:
            logger.warning(f"Could not enable compression: {e}")
            return base_retriever
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Optional override for number of documents
            
        Returns:
            List of relevant documents
        """
        try:
            if top_k:
                # Update retriever's k parameter if needed
                if hasattr(self.retriever, 'search_kwargs'):
                    self.retriever.search_kwargs['k'] = top_k
            
            results = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(results)} documents for query")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve documents with relevance scores
        
        Args:
            query: Query text
            top_k: Optional override for number of documents
            
        Returns:
            List of (document, score) tuples
        """
        # This is a simplified version; actual implementation depends on retriever type
        documents = self.retrieve(query, top_k)
        
        # If we can't get actual scores, return with placeholder scores
        return [(doc, 1.0) for doc in documents]


class CohereReranker:
    """
    Reranker using Cohere's rerank API for improved relevance
    """
    
    def __init__(self, api_key: str, model: str = "rerank-english-v2.0"):
        """
        Initialize Cohere reranker
        
        Args:
            api_key: Cohere API key
            model: Rerank model to use
        """
        try:
            import cohere
            self.client = cohere.Client(api_key)
            self.model = model
            logger.info(f"Initialized Cohere reranker with model: {model}")
        except ImportError:
            logger.error("Cohere package not installed")
            raise
        except Exception as e:
            logger.error(f"Error initializing Cohere reranker: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        Rerank documents using Cohere's rerank API
        
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        try:
            # Extract text content from documents
            texts = [doc.page_content for doc in documents]
            
            # Call Cohere rerank API
            results = self.client.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_n=top_k
            )
            
            # Reorder documents based on rerank scores
            reranked_docs = []
            for result in results.results:
                doc = documents[result.index]
                doc.metadata['rerank_score'] = result.relevance_score
                reranked_docs.append(doc)
            
            logger.info(f"Reranked {len(documents)} documents to top {top_k}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents[:top_k]

