"""
Multimodal RAG Pipeline - Supports both text and images
Leverages GPT-4 Vision and CLIP embeddings for enhanced understanding
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import logging
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from .config import RAGConfig
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retrieval import AdvancedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalDocument:
    """
    Represents a document that can contain both text and images
    """
    def __init__(
        self, 
        text: str = "", 
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        metadata: Optional[Dict] = None
    ):
        self.text = text
        self.image_path = image_path
        self.image_data = image_data
        self.metadata = metadata or {}
        
    def has_image(self) -> bool:
        return self.image_path is not None or self.image_data is not None
    
    def get_image_base64(self) -> Optional[str]:
        """Get base64 encoded image for API calls"""
        try:
            if self.image_data:
                return base64.b64encode(self.image_data).decode('utf-8')
            elif self.image_path:
                with open(self.image_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
        return None


class MultimodalEmbeddings:
    """
    Generate embeddings for both text and images using CLIP
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.dimension = 512  # CLIP embedding dimension
            
            logger.info(f"Initialized CLIP model on {self.device}")
        except ImportError:
            logger.warning("CLIP not available, install transformers and torch")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.model:
            raise ValueError("CLIP model not initialized")
        
        import torch
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            embedding = text_features.cpu().numpy()[0]
        
        return embedding
    
    def embed_image(self, image: Union[str, Image.Image, bytes]) -> np.ndarray:
        """Generate embedding for image"""
        if not self.model:
            raise ValueError("CLIP model not initialized")
        
        import torch
        
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image)).convert('RGB')
        else:
            pil_image = image
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.cpu().numpy()[0]
        
        return embedding
    
    def embed_multimodal(
        self, 
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        text_weight: float = 0.5
    ) -> np.ndarray:
        """
        Generate combined embedding for text and image
        
        Args:
            text: Text content
            image: Image content
            text_weight: Weight for text embedding (0-1)
        """
        if text and image:
            text_emb = self.embed_text(text)
            image_emb = self.embed_image(image)
            # Weighted combination
            combined = text_weight * text_emb + (1 - text_weight) * image_emb
            # Normalize
            combined = combined / np.linalg.norm(combined)
            return combined
        elif text:
            return self.embed_text(text)
        elif image:
            return self.embed_image(image)
        else:
            raise ValueError("At least one of text or image must be provided")


class MultimodalRAGPipeline:
    """
    Enhanced RAG pipeline with multimodal capabilities
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize Multimodal RAG pipeline
        
        Args:
            config: RAG configuration object
        """
        self.config = config or RAGConfig()
        
        # Initialize text components
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy="recursive"
        )
        
        # Initialize multimodal embeddings
        try:
            self.multimodal_embeddings = MultimodalEmbeddings()
            logger.info("Multimodal embeddings initialized")
        except Exception as e:
            logger.warning(f"Could not initialize multimodal embeddings: {e}")
            self.multimodal_embeddings = None
        
        # Initialize standard text embeddings as fallback
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        
        # Initialize vector store
        self.vector_store_manager = VectorStoreManager(
            embeddings=self.embedding_manager.get_embeddings_instance(),
            store_type=self.config.vector_store_type,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory
        )
        
        self.documents = []
        self.multimodal_documents = []
        self.retriever = None
        
        logger.info("Multimodal RAG Pipeline initialized")
    
    def ingest_documents(
        self,
        source_path: str,
        create_new: bool = True
    ) -> None:
        """
        Ingest text documents from a file or directory (standard RAG ingestion)
        
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
    
    def ingest_multimodal_documents(
        self,
        documents: List[MultimodalDocument],
        create_new: bool = True
    ) -> None:
        """
        Ingest multimodal documents (text + images)
        
        Args:
            documents: List of MultimodalDocument objects
            create_new: Whether to create new vector store
        """
        try:
            self.multimodal_documents = documents
            
            # Convert to LangChain documents for vector store
            langchain_docs = []
            for idx, doc in enumerate(documents):
                # Create metadata
                metadata = {
                    **doc.metadata,
                    'doc_id': idx,
                    'has_image': doc.has_image(),
                    'image_path': doc.image_path
                }
                
                # Create document
                langchain_doc = Document(
                    page_content=doc.text,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)
            
            self.documents = langchain_docs
            
            # Create or update vector store
            if create_new:
                logger.info("Creating vector store for multimodal documents...")
                self.vector_store_manager.create_vector_store(langchain_docs)
            else:
                logger.info("Adding to existing vector store...")
                self.vector_store_manager.load_vector_store()
                self.vector_store_manager.add_documents(langchain_docs)
            
            # Initialize retriever
            self._initialize_retriever()
            
            logger.info(f"Successfully ingested {len(documents)} multimodal documents")
            
        except Exception as e:
            logger.error(f"Error ingesting multimodal documents: {e}")
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
    
    def query(
        self,
        question: str,
        return_source_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Standard text-only query (compatible with original RAG pipeline)
        
        Args:
            question: Question to ask
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with 'answer' and optionally 'source_documents'
        """
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized. Load documents first.")
            
            # Retrieve relevant documents
            documents = self.retriever.retrieve(question, top_k=self.config.top_k_retrieval)
            
            # Use text-only query
            answer = self._query_text_only(question, documents)
            
            response = {
                "answer": answer,
            }
            
            if return_source_documents:
                response["source_documents"] = documents
            
            return response
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            raise
    
    def query_multimodal(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Union[str, bytes]] = None,
        return_source_documents: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system with text and/or image
        
        Args:
            query_text: Text query
            query_image: Image query (path or bytes)
            return_source_documents: Whether to return source documents
            
        Returns:
            Dictionary with 'answer' and optionally 'source_documents'
        """
        try:
            if not query_text and not query_image:
                raise ValueError("At least one of query_text or query_image must be provided")
            
            # Retrieve relevant documents (using text retrieval for now)
            if query_text:
                documents = self.retriever.retrieve(query_text, top_k=self.config.top_k_retrieval)
            else:
                # For image-only queries, use a generic search
                documents = self.retriever.retrieve("visual content", top_k=self.config.top_k_retrieval)
            
            # Initialize GPT-4 Vision if image is provided
            use_vision = query_image is not None
            
            if use_vision:
                # Use GPT-4 Vision for multimodal understanding
                answer = self._query_with_vision(query_text, query_image, documents)
            else:
                # Standard text query
                answer = self._query_text_only(query_text, documents)
            
            response = {
                "answer": answer,
                "multimodal": use_vision
            }
            
            if return_source_documents:
                response["source_documents"] = documents
            
            return response
            
        except Exception as e:
            logger.error(f"Error during multimodal query: {e}")
            raise
    
    def _query_with_vision(
        self,
        query_text: Optional[str],
        query_image: Union[str, bytes],
        context_documents: List[Document]
    ) -> str:
        """
        Query using GPT-4 Vision with context
        
        Args:
            query_text: Text query
            query_image: Image query
            context_documents: Retrieved context documents
        """
        try:
            # Encode image
            if isinstance(query_image, str):
                with open(query_image, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            else:
                image_data = base64.b64encode(query_image).decode('utf-8')
            
            # Prepare context from documents
            context = "\n\n".join([doc.page_content for doc in context_documents[:3]])
            
            # Initialize GPT-4 Vision
            llm = ChatOpenAI(
                model_name="gpt-4o",  # GPT-4o has vision capabilities
                temperature=self.config.temperature,
                openai_api_key=self.config.openai_api_key,
                max_tokens=self.config.max_tokens
            )
            
            # Create multimodal prompt
            prompt = f"""You are an intelligent multimodal AI assistant that can understand both images and text. 

Context from knowledge base:
{context if context else "No relevant context found."}

User's question: {query_text if query_text else "Please analyze this image."}

The user has provided an image. Analyze the image carefully and provide a comprehensive answer that:
1. Describes what you see in the image
2. Relates the image to the user's question
3. Uses the provided context when relevant
4. Provides insights and explanations

Answer:"""
            
            # For GPT-4 Vision, we need to use the messages API
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens
            )
            
            answer = response.choices[0].message.content
            return f"🖼️ **Multimodal Analysis:**\n\n{answer}"
            
        except Exception as e:
            logger.error(f"Error in vision query: {e}")
            return f"Error processing image query: {str(e)}"
    
    def _query_text_only(self, query_text: str, documents: List[Document]) -> str:
        """Standard text-only query"""
        if not documents:
            llm = ChatOpenAI(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                openai_api_key=self.config.openai_api_key,
                max_tokens=self.config.max_tokens
            )
            
            prompt = f"Answer this question using your general knowledge: {query_text}"
            answer = llm.predict(prompt)
            return f"ℹ️ **Note:** Answer from general knowledge:\n\n{answer}"
        
        # Use retrieved documents
        context = "\n\n".join([doc.page_content for doc in documents[:3]])
        
        llm = ChatOpenAI(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            openai_api_key=self.config.openai_api_key,
            max_tokens=self.config.max_tokens
        )
        
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query_text}

Answer:"""
        
        answer = llm.predict(prompt)
        return answer
    
    def get_embedding_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing embeddings (for frontend visualization)
        """
        try:
            if not self.documents:
                return {"embeddings": [], "labels": [], "has_images": []}
            
            # Get embeddings from vector store
            embeddings_data = []
            labels = []
            has_images = []
            
            for doc in self.documents[:100]:  # Limit to 100 for visualization
                labels.append(doc.page_content[:50] + "...")
                has_images.append(doc.metadata.get('has_image', False))
            
            return {
                "embeddings": [],  # We'll compute these on demand
                "labels": labels,
                "has_images": has_images,
                "num_documents": len(self.documents),
                "multimodal_enabled": self.multimodal_embeddings is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            return {"error": str(e)}
    
    def analyze_image(self, image: Union[str, bytes]) -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision
        
        Args:
            image: Image path or bytes
            
        Returns:
            Analysis results
        """
        try:
            # Encode image
            if isinstance(image, str):
                with open(image, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            else:
                image_data = base64.b64encode(image).decode('utf-8')
            
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Analyze this image in detail. Describe what you see, identify key objects, colors, composition, and any notable features."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            # Get embedding if available
            embedding = None
            if self.multimodal_embeddings:
                embedding = self.multimodal_embeddings.embed_image(image).tolist()
            
            return {
                "analysis": analysis,
                "embedding": embedding,
                "embedding_dimension": len(embedding) if embedding else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the multimodal RAG system"""
        multimodal_docs = sum(1 for doc in self.documents if doc.metadata.get('has_image', False))
        
        return {
            "num_documents": len(self.documents),
            "num_multimodal_documents": multimodal_docs,
            "num_text_only_documents": len(self.documents) - multimodal_docs,
            "multimodal_embeddings_available": self.multimodal_embeddings is not None,
            "embedding_model": self.config.embedding_model,
            "llm_model": self.config.llm_model,
            "vision_enabled": True,
            "clip_available": self.multimodal_embeddings is not None
        }

