"""
Production-Ready RAG API Server with Authentication and Caching

This server integrates:
- API Key Authentication (from practice_auth.py)
- Query Caching (from practice_caching.py)
- RAG Pipeline with full features
- Query history tracking
- Rate limiting
- Performance monitoring

Run with: uvicorn api_server_production:app --reload --port 8000
API docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import sqlite3
import json
from pathlib import Path
import logging
import time

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig
from practice_auth import APIKeyManager, APIKey, APIKeyValidation
from practice_caching import QueryCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API - Production",
    description="Production-ready RAG API with authentication, caching, and monitoring",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_pipeline: Optional[RAGPipeline] = None
api_key_manager: Optional[APIKeyManager] = None
query_cache: Optional[QueryCache] = None

# Database path
DB_PATH = Path("./data/query_history.db")

# API Key header configuration
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    use_cache: bool = Field(True, description="Whether to use cache")
    return_sources: bool = Field(True, description="Whether to return source documents")


class QueryResponse(BaseModel):
    """Response model for query results"""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default=[], description="Source documents used")
    query_id: int = Field(..., description="Unique query ID for tracking")
    timestamp: datetime = Field(..., description="Query timestamp")
    from_cache: bool = Field(False, description="Whether result came from cache")
    response_time_ms: float = Field(..., description="Response time in milliseconds")


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key"""
    name: str = Field(..., description="Human-readable name for the key")
    user_id: str = Field(..., description="User ID who owns this key")
    rate_limit: int = Field(60, description="Requests per minute", ge=1, le=1000)
    expires_in_days: Optional[int] = Field(None, description="Days until expiration")


class CreateAPIKeyResponse(BaseModel):
    """Response after creating API key"""
    api_key: str = Field(..., description="The API key (shown only once!)")
    key_id: int = Field(..., description="Key ID for management")
    key_prefix: str = Field(..., description="Key prefix for identification")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    message: str = Field(..., description="Important notice")


class APIKeyInfo(BaseModel):
    """Information about an API key"""
    key_id: int
    name: str
    key_prefix: str
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    rate_limit: int
    usage_count: int


class CacheStats(BaseModel):
    """Cache statistics"""
    total_requests: int
    hits: int
    misses: int
    hit_rate: float
    current_size: int
    max_size: int
    evictions: int


class SystemHealth(BaseModel):
    """System health status"""
    status: str
    rag_initialized: bool
    auth_initialized: bool
    cache_initialized: bool
    cache_stats: Optional[CacheStats]
    timestamp: datetime


# ============================================================================
# Database Functions
# ============================================================================

def init_database():
    """Initialize SQLite database with query history table"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            user_id TEXT,
            api_key_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time_ms REAL,
            from_cache BOOLEAN
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON query_history(timestamp DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id 
        ON query_history(user_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_api_key_id 
        ON query_history(api_key_id)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_query_to_history(
    question: str,
    answer: str,
    sources: List[str],
    user_id: str,
    api_key_id: int,
    response_time_ms: float,
    from_cache: bool
) -> int:
    """Save query to history database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO query_history 
        (question, answer, sources, user_id, api_key_id, response_time_ms, from_cache)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        json.dumps(sources),
        user_id,
        api_key_id,
        response_time_ms,
        from_cache
    ))
    
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return query_id


# ============================================================================
# Authentication Dependency
# ============================================================================

async def validate_api_key_dependency(api_key: str = Security(api_key_header)) -> APIKey:
    """
    Dependency to validate API keys on protected endpoints
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        APIKey object if valid
        
    Raises:
        HTTPException: If API key is invalid, expired, or rate limited
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header."
        )
    
    if not api_key_manager:
        raise HTTPException(
            status_code=503,
            detail="Authentication system not initialized"
        )
    
    validation = api_key_manager.validate_api_key(api_key)
    
    if not validation.is_valid:
        raise HTTPException(
            status_code=401,
            detail=validation.error_message
        )
    
    return validation.api_key


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all systems on startup"""
    global rag_pipeline, api_key_manager, query_cache
    
    logger.info("Starting up production RAG API server...")
    
    # Initialize database
    init_database()
    
    # Initialize API Key Manager
    try:
        api_key_manager = APIKeyManager(db_path="./data/api_keys.db")
        logger.info("API Key Manager initialized")
    except Exception as e:
        logger.error(f"Error initializing API Key Manager: {e}")
        raise
    
    # Initialize Query Cache
    try:
        query_cache = QueryCache(max_size=1000, ttl_seconds=3600)
        logger.info("Query Cache initialized (max_size=1000, ttl=1h)")
    except Exception as e:
        logger.error(f"Error initializing Query Cache: {e}")
        raise
    
    # Initialize RAG pipeline with intelligent configuration
    try:
        config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-large",  # OpenAI embeddings for better semantic understanding
            llm_model="gpt-4o",  # GPT-4o for most intelligent responses
            temperature=0.3,  # Lower temperature for more accurate, focused answers
            top_k_retrieval=5,
            use_hybrid_search=True,  # Enable hybrid search for better retrieval
            use_reranking=False,  # Disable reranking unless Cohere API key is available
            use_contextual_compression=True,  # Enable compression for better context
            max_tokens=4000,  # Increased for detailed responses
            collection_name='production_rag_documents'
        )
        
        rag_pipeline = RAGPipeline(config)
        
        # Try to load existing documents
        try:
            rag_pipeline.load_existing_store()
            logger.info("Loaded existing vector store")
        except:
            doc_path = Path("documents")
            if doc_path.exists():
                rag_pipeline.ingest_documents("documents", create_new=True)
                logger.info("Ingested documents")
            else:
                logger.warning("No documents found. Add documents to ./documents/")
    
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        # Don't raise - allow server to start even without documents
    
    logger.info("✅ Production RAG API server started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down production RAG API server...")
    if query_cache:
        stats = query_cache.get_stats()
        logger.info(f"Final cache stats: {stats}")


# ============================================================================
# Public Endpoints (No Auth Required)
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG System API - Production",
        "version": "2.0.0",
        "features": [
            "API Key Authentication",
            "Query Caching",
            "Rate Limiting",
            "Query History",
            "Performance Monitoring"
        ],
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "create_key": "POST /api/keys/create",
            "query": "POST /api/query (requires API key)",
            "cache_stats": "GET /api/cache/stats (requires API key)"
        }
    }


@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Health check endpoint"""
    cache_stats_data = None
    if query_cache:
        stats = query_cache.get_stats()
        cache_stats_data = CacheStats(
            total_requests=stats['total_requests'],
            hits=stats['hits'],
            misses=stats['misses'],
            hit_rate=round(stats['hit_rate'], 2),
            current_size=stats['current_size'],
            max_size=query_cache.max_size,
            evictions=query_cache.stats.get('evictions', 0)
        )
    
    return SystemHealth(
        status="healthy",
        rag_initialized=rag_pipeline is not None,
        auth_initialized=api_key_manager is not None,
        cache_initialized=query_cache is not None,
        cache_stats=cache_stats_data,
        timestamp=datetime.now(timezone.utc)
    )


@app.post("/api/keys/create", response_model=CreateAPIKeyResponse)
async def create_api_key(request: CreateAPIKeyRequest):
    """
    Create a new API key
    
    Note: In production, this should be protected with admin authentication
    """
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="Auth system not initialized")
    
    try:
        raw_key, api_key = api_key_manager.generate_api_key(
            name=request.name,
            user_id=request.user_id,
            rate_limit=request.rate_limit,
            expires_in_days=request.expires_in_days
        )
        
        logger.info(f"Created API key for user: {request.user_id}")
        
        return CreateAPIKeyResponse(
            api_key=raw_key,
            key_id=api_key.key_id,
            key_prefix=api_key.key_prefix,
            expires_at=api_key.expires_at,
            message="⚠️ IMPORTANT: Save this API key now! It won't be shown again."
        )
        
    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating API key: {str(e)}")


# ============================================================================
# Protected Endpoints (Auth Required)
# ============================================================================

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    api_key: APIKey = Depends(validate_api_key_dependency)
):
    """
    Query the RAG system with authentication and caching
    
    Requires: X-API-Key header with valid API key
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    start_time = time.time()
    from_cache = False
    
    try:
        # Try cache first if enabled
        cached_result = None
        if request.use_cache and query_cache:
            cached_result = query_cache.get(
                request.question,
                user_id=api_key.user_id
            )
        
        if cached_result:
            # Cache hit!
            from_cache = True
            answer = cached_result['answer']
            sources = cached_result.get('sources', [])
            logger.info(f"Cache HIT for user {api_key.user_id}")
        else:
            # Cache miss - query RAG pipeline
            logger.info(f"Cache MISS for user {api_key.user_id}")
            
            # Option 1: Simple (current) - just return document text
            if False:  # Change to True to use this mode
                documents = rag_pipeline.retrieve(request.question)
                if not documents:
                    answer = "I couldn't find relevant information to answer your question."
                    sources = []
                else:
                    answer = f"Based on the documents: {documents[0].page_content}"
                    sources = [doc.page_content[:200] + "..." for doc in documents[:3]]
            
            # Option 2: Full RAG with LLM (requires OpenAI API key)
            else:
                try:
                    result = rag_pipeline.query(
                        request.question,
                        return_source_documents=True
                    )
                    answer = result['answer']
                    sources = [doc.page_content[:200] + "..." for doc in result.get('source_documents', [])[:3]]
                except Exception as e:
                    logger.error(f"LLM query failed: {e}, falling back to simple retrieval")
                    documents = rag_pipeline.retrieve(request.question)
                    if not documents:
                        answer = "I couldn't find relevant information to answer your question."
                        sources = []
                    else:
                        answer = f"Based on the documents: {documents[0].page_content}"
                        sources = [doc.page_content[:200] + "..." for doc in documents[:3]]
            
            # Store in cache if enabled
            if request.use_cache and query_cache:
                cache_data = {
                    'answer': answer,
                    'sources': sources
                }
                query_cache.set(
                    request.question,
                    cache_data,
                    user_id=api_key.user_id
                )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Record usage for rate limiting and analytics
        api_key_manager.record_usage(api_key, "/api/query", True)
        
        # Save to history
        query_id = save_query_to_history(
            question=request.question,
            answer=answer,
            sources=sources,
            user_id=api_key.user_id,
            api_key_id=api_key.key_id,
            response_time_ms=response_time_ms,
            from_cache=from_cache
        )
        
        logger.info(
            f"Query processed: user={api_key.user_id}, "
            f"time={response_time_ms:.2f}ms, cache={from_cache}"
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources if request.return_sources else [],
            query_id=query_id,
            timestamp=datetime.now(timezone.utc),
            from_cache=from_cache,
            response_time_ms=round(response_time_ms, 2)
        )
        
    except Exception as e:
        # Record failed usage
        api_key_manager.record_usage(api_key, "/api/query", False)
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/cache/stats", response_model=CacheStats)
async def get_cache_stats(api_key: APIKey = Depends(validate_api_key_dependency)):
    """Get cache statistics (requires API key)"""
    if not query_cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    # Cleanup expired entries before getting stats
    removed = query_cache.cleanup_expired()
    if removed > 0:
        logger.info(f"Cleaned up {removed} expired cache entries")
    
    stats = query_cache.get_stats()
    
    return CacheStats(
        total_requests=stats['total_requests'],
        hits=stats['hits'],
        misses=stats['misses'],
        hit_rate=round(stats['hit_rate'], 2),
        current_size=stats['current_size'],
        max_size=query_cache.max_size,
        evictions=query_cache.stats.get('evictions', 0)
    )


@app.post("/api/cache/invalidate")
async def invalidate_cache(
    query: Optional[str] = None,
    api_key: APIKey = Depends(validate_api_key_dependency)
):
    """
    Invalidate cache entries
    
    Args:
        query: Specific query to invalidate, or None to clear all
    """
    if not query_cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    try:
        query_cache.invalidate(query)
        message = "All cache cleared" if query is None else f"Cache cleared for query: {query}"
        logger.info(f"Cache invalidated by user {api_key.user_id}: {message}")
        return {"message": message}
        
    except Exception as e:
        logger.error(f"Error invalidating cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {str(e)}")


@app.get("/api/keys/info", response_model=APIKeyInfo)
async def get_api_key_info(api_key: APIKey = Depends(validate_api_key_dependency)):
    """Get information about the current API key"""
    return APIKeyInfo(
        key_id=api_key.key_id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        user_id=api_key.user_id,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        is_active=api_key.is_active,
        rate_limit=api_key.rate_limit,
        usage_count=api_key.usage_count
    )


@app.get("/api/keys/usage")
async def get_api_key_usage(api_key: APIKey = Depends(validate_api_key_dependency)):
    """Get usage statistics for the current API key"""
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="Auth system not initialized")
    
    try:
        stats = api_key_manager.get_usage_stats(api_key.key_id)
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving usage stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.post("/api/keys/revoke/{key_id}")
async def revoke_api_key(
    key_id: int,
    api_key: APIKey = Depends(validate_api_key_dependency)
):
    """
    Revoke an API key
    
    Note: Users can only revoke their own keys
    """
    if not api_key_manager:
        raise HTTPException(status_code=503, detail="Auth system not initialized")
    
    # Check if user owns this key
    if api_key.key_id != key_id and api_key.user_id != "admin":
        raise HTTPException(
            status_code=403,
            detail="You can only revoke your own keys"
        )
    
    try:
        success = api_key_manager.revoke_api_key(key_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        
        logger.info(f"API key {key_id} revoked by user {api_key.user_id}")
        return {"message": f"API key {key_id} revoked successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key: {e}")
        raise HTTPException(status_code=500, detail=f"Error revoking key: {str(e)}")


@app.get("/api/history")
async def get_query_history(
    limit: int = Query(default=10, ge=1, le=100),
    api_key: APIKey = Depends(validate_api_key_dependency)
):
    """Get query history for the current user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM query_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (api_key.user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row['id'],
                'question': row['question'],
                'answer': row['answer'],
                'timestamp': row['timestamp'],
                'response_time_ms': row['response_time_ms'],
                'from_cache': bool(row['from_cache'])
            })
        
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

