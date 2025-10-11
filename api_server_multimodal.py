"""
Enhanced API Server with Multimodal Capabilities
Supports text, images, and combined queries with visualization endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Security, UploadFile, File, Form
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import os
from pathlib import Path
import base64
from io import BytesIO

# Import RAG components
from src.config import RAGConfig
from src.multimodal_rag import MultimodalRAGPipeline, MultimodalDocument
from practice_auth import APIKeyManager, APIKey

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎨 Multimodal RAG API",
    description="Advanced RAG system with vision and text capabilities",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
api_key_manager = APIKeyManager()
rag_pipeline: Optional[MultimodalRAGPipeline] = None

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)) -> APIKey:
    """Dependency to validate API keys"""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    validation = api_key_manager.validate_api_key(api_key)
    
    if not validation.is_valid:
        raise HTTPException(status_code=401, detail=validation.error_message)
    
    return validation.api_key


# Request/Response Models
class QueryRequest(BaseModel):
    """Text query request"""
    question: str = Field(..., description="The question to ask")
    return_sources: bool = Field(True, description="Whether to return source documents")
    use_cache: bool = Field(True, description="Whether to use cached results")


class QueryResponse(BaseModel):
    """Query response"""
    answer: str
    sources: Optional[List[str]] = None
    multimodal: bool = False
    from_cache: bool = False
    timestamp: str


class ImageAnalysisResponse(BaseModel):
    """Image analysis response"""
    analysis: str
    embedding: Optional[List[float]] = None
    embedding_dimension: Optional[int] = None


class MultimodalQueryResponse(BaseModel):
    """Multimodal query response"""
    answer: str
    multimodal: bool = True
    sources: Optional[List[str]] = None
    timestamp: str


class SystemStats(BaseModel):
    """System statistics"""
    num_documents: int
    num_multimodal_documents: int
    multimodal_embeddings_available: bool
    vision_enabled: bool
    llm_model: str


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    try:
        logger.info("Initializing Multimodal RAG pipeline...")
        config = RAGConfig()
        rag_pipeline = MultimodalRAGPipeline(config)
        
        # Try to load existing vector store
        try:
            rag_pipeline.load_existing_store()
            logger.info("✅ Loaded existing vector store")
        except Exception as e:
            logger.warning(f"No existing vector store found: {e}")
            logger.info("⚠️  RAG pipeline ready but no documents loaded")
            logger.info("   System will use general knowledge for queries")
        
        logger.info("✅ Multimodal RAG API Server started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🎨 Multimodal RAG API Server",
        "version": "2.0.0",
        "features": [
            "Text queries",
            "Image analysis",
            "Multimodal queries (text + image)",
            "CLIP embeddings",
            "GPT-4o Vision",
            "Visualizations"
        ],
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_initialized": rag_pipeline is not None,
        "multimodal_enabled": rag_pipeline.multimodal_embeddings is not None if rag_pipeline else False,
        "vision_enabled": True
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: APIKey = Depends(validate_api_key)
) -> QueryResponse:
    """
    Standard text query endpoint
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Record API usage
        api_key_manager.record_usage(api_key, "/api/query", True)
        
        # Process query
        result = rag_pipeline.query(
            question=request.question,
            return_source_documents=request.return_sources
        )
        
        # Extract sources
        sources = []
        if request.return_sources and "source_documents" in result:
            sources = [
                doc.metadata.get('source', 'Unknown')
                for doc in result["source_documents"]
            ]
        
        return QueryResponse(
            answer=result["answer"],
            sources=sources if request.return_sources else None,
            multimodal=False,
            from_cache=False,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        api_key_manager.record_usage(api_key, "/api/query", False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multimodal/query", response_model=MultimodalQueryResponse)
async def multimodal_query(
    image: UploadFile = File(...),
    query_text: Optional[str] = Form(None),
    api_key: APIKey = Depends(validate_api_key)
) -> MultimodalQueryResponse:
    """
    Multimodal query endpoint - supports text + image or image only
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Record API usage
        api_key_manager.record_usage(api_key, "/api/multimodal/query", True)
        
        # Read image data
        image_data = await image.read()
        
        # Process multimodal query
        result = rag_pipeline.query_multimodal(
            query_text=query_text,
            query_image=image_data,
            return_source_documents=True
        )
        
        # Extract sources
        sources = []
        if "source_documents" in result:
            sources = [
                doc.metadata.get('source', 'Unknown')
                for doc in result["source_documents"]
            ]
        
        return MultimodalQueryResponse(
            answer=result["answer"],
            multimodal=True,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing multimodal query: {e}")
        api_key_manager.record_usage(api_key, "/api/multimodal/query", False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multimodal/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    api_key: APIKey = Depends(validate_api_key)
) -> ImageAnalysisResponse:
    """
    Analyze an image using GPT-4 Vision and get CLIP embedding
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Record API usage
        api_key_manager.record_usage(api_key, "/api/multimodal/analyze-image", True)
        
        # Read image data
        image_data = await image.read()
        
        # Analyze image
        result = rag_pipeline.analyze_image(image_data)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return ImageAnalysisResponse(
            analysis=result.get("analysis", ""),
            embedding=result.get("embedding"),
            embedding_dimension=result.get("embedding_dimension")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        api_key_manager.record_usage(api_key, "/api/multimodal/analyze-image", False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats)
async def get_stats(api_key: APIKey = Depends(validate_api_key)) -> SystemStats:
    """
    Get system statistics
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        stats = rag_pipeline.get_stats()
        return SystemStats(
            num_documents=stats["num_documents"],
            num_multimodal_documents=stats["num_multimodal_documents"],
            multimodal_embeddings_available=stats["multimodal_embeddings_available"],
            vision_enabled=stats["vision_enabled"],
            llm_model=stats["llm_model"]
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/visualization/embeddings")
async def get_embedding_visualization(
    api_key: APIKey = Depends(validate_api_key)
) -> Dict[str, Any]:
    """
    Get embedding visualization data
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        viz_data = rag_pipeline.get_embedding_visualization_data()
        return viz_data
        
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API Key Management Endpoints

@app.post("/api/keys")
async def create_api_key(
    name: str,
    user_id: str,
    rate_limit: int = 60,
    expires_in_days: Optional[int] = 365
):
    """
    Create a new API key (admin endpoint)
    """
    try:
        raw_key, api_key_obj = api_key_manager.generate_api_key(
            name=name,
            user_id=user_id,
            rate_limit=rate_limit,
            expires_in_days=expires_in_days
        )
        
        return {
            "api_key": raw_key,
            "key_id": api_key_obj.key_id,
            "key_prefix": api_key_obj.key_prefix,
            "expires_at": api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None,
            "note": "⚠️ Save this key securely - it won't be shown again!"
        }
        
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/keys/info")
async def get_key_info(api_key: APIKey = Depends(validate_api_key)):
    """
    Get information about the current API key
    """
    return {
        "key_id": api_key.key_id,
        "name": api_key.name,
        "user_id": api_key.user_id,
        "key_prefix": api_key.key_prefix,
        "created_at": api_key.created_at.isoformat(),
        "rate_limit": api_key.rate_limit,
        "usage_count": api_key.usage_count,
        "is_active": api_key.is_active
    }


@app.get("/api/keys/{key_id}/stats")
async def get_key_stats(
    key_id: int,
    api_key: APIKey = Depends(validate_api_key)
):
    """
    Get usage statistics for an API key
    """
    try:
        stats = api_key_manager.get_usage_stats(key_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting key stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Ingestion Endpoint (for testing)

@app.post("/api/ingest/text")
async def ingest_text_documents(
    source_path: str,
    create_new: bool = True,
    api_key: APIKey = Depends(validate_api_key)
):
    """
    Ingest text documents from a path
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.ingest_documents(source_path, create_new)
        
        return {
            "status": "success",
            "message": f"Ingested documents from {source_path}",
            "num_documents": len(rag_pipeline.documents)
        }
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Multimodal RAG API Server...")
    print("📊 Features: Text + Vision + CLIP Embeddings")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("🎨 Frontend: Open frontend_multimodal.html")
    
    uvicorn.run(
        "api_server_multimodal:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

