"""
REST API Server for RAG System with Query History Tracking

This is a complete backend implementation that demonstrates:
- RESTful API design with FastAPI
- Database integration with SQLite
- Query history tracking and analytics
- Error handling and validation
- CORS configuration
- API documentation (automatic with FastAPI)

Run with: uvicorn api_server:app --reload
API docs available at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlite3
import json
from pathlib import Path
import logging

from src.rag_pipeline import RAGPipeline
from src.config import RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="REST API for RAG document Q&A with query history tracking",
    version="1.0.0"
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

# Database path
DB_PATH = Path("./data/query_history.db")


# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for querying the RAG system"""
    question: str = Field(..., description="The question to ask", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")


class QueryResponse(BaseModel):
    """Response model for query results"""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(default=[], description="Source documents used")
    query_id: int = Field(..., description="Unique query ID for tracking")
    timestamp: datetime = Field(..., description="Query timestamp")


class QueryHistoryItem(BaseModel):
    """Model for query history item"""
    id: int
    question: str
    answer: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    response_time_ms: float


class SystemStats(BaseModel):
    """Model for system statistics"""
    total_queries: int
    total_documents: int
    avg_response_time_ms: float
    most_common_topics: List[Dict[str, Any]]


# ============================================================================
# Database Functions
# ============================================================================

def init_database():
    """Initialize SQLite database with query history table"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create query history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,
            user_id TEXT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time_ms REAL
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON query_history(timestamp DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id 
        ON query_history(user_id)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_query_to_history(
    question: str,
    answer: str,
    sources: List[str],
    user_id: Optional[str],
    session_id: Optional[str],
    response_time_ms: float
) -> int:
    """
    Save a query and its response to the database
    
    Args:
        question: The user's question
        answer: The generated answer
        sources: List of source documents
        user_id: Optional user identifier
        session_id: Optional session identifier
        response_time_ms: Time taken to generate response
        
    Returns:
        The ID of the inserted query
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO query_history 
        (question, answer, sources, user_id, session_id, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        question,
        answer,
        json.dumps(sources),  # Store sources as JSON
        user_id,
        session_id,
        response_time_ms
    ))
    
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return query_id


def get_query_history(
    limit: int = 10,
    user_id: Optional[str] = None
) -> List[QueryHistoryItem]:
    """
    Retrieve query history from database
    
    Args:
        limit: Maximum number of queries to return
        user_id: Optional filter by user ID
        
    Returns:
        List of query history items
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute("""
            SELECT * FROM query_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM query_history
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to Pydantic models
    history = []
    for row in rows:
        history.append(QueryHistoryItem(
            id=row['id'],
            question=row['question'],
            answer=row['answer'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            response_time_ms=row['response_time_ms']
        ))
    
    return history


def get_system_stats() -> SystemStats:
    """
    Get system statistics from database
    
    Returns:
        SystemStats object with analytics data
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total queries
    cursor.execute("SELECT COUNT(*) FROM query_history")
    total_queries = cursor.fetchone()[0]
    
    # Average response time
    cursor.execute("SELECT AVG(response_time_ms) FROM query_history")
    avg_time = cursor.fetchone()[0] or 0.0
    
    # Most common question keywords (simple implementation)
    cursor.execute("""
        SELECT question, COUNT(*) as count
        FROM query_history
        GROUP BY question
        ORDER BY count DESC
        LIMIT 5
    """)
    common_questions = [
        {"question": row[0], "count": row[1]}
        for row in cursor.fetchall()
    ]
    
    conn.close()
    
    # Get document count from RAG pipeline
    doc_count = 0
    if rag_pipeline:
        stats = rag_pipeline.get_stats()
        doc_count = stats.get('num_documents', 0)
    
    return SystemStats(
        total_queries=total_queries,
        total_documents=doc_count,
        avg_response_time_ms=round(avg_time, 2),
        most_common_topics=common_questions
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global rag_pipeline
    
    # Initialize database
    init_database()
    
    # Initialize RAG pipeline
    try:
        config = RAGConfig(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            top_k_retrieval=5,
            use_hybrid_search=False,
            use_reranking=False,
            collection_name='api_rag_documents'
        )
        
        rag_pipeline = RAGPipeline(config)
        
        # Try to load existing documents
        try:
            rag_pipeline.load_existing_store()
            logger.info("Loaded existing vector store")
        except:
            # Ingest documents if available
            doc_path = Path("documents")
            if doc_path.exists():
                rag_pipeline.ingest_documents("documents", create_new=True)
                logger.info("Ingested documents")
    
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "query": "/api/query",
            "history": "/api/history",
            "stats": "/api/stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question
    
    Args:
        request: QueryRequest with question and optional user/session IDs
        
    Returns:
        QueryResponse with answer, sources, and query ID
        
    Raises:
        HTTPException: If RAG system is not initialized or query fails
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Measure response time
        start_time = datetime.now()
        
        # Retrieve documents
        documents = rag_pipeline.retrieve(request.question)
        
        if not documents:
            answer = "I couldn't find relevant information to answer your question."
            sources = []
        else:
            # Get the most relevant document as answer
            answer = f"Based on the documents: {documents[0].page_content}"
            sources = [doc.page_content[:200] + "..." for doc in documents[:3]]
        
        # Calculate response time
        end_time = datetime.now()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Save to database
        query_id = save_query_to_history(
            question=request.question,
            answer=answer,
            sources=sources,
            user_id=request.user_id,
            session_id=request.session_id,
            response_time_ms=response_time_ms
        )
        
        logger.info(f"Query processed: ID={query_id}, Time={response_time_ms:.2f}ms")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_id=query_id,
            timestamp=end_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/history", response_model=List[QueryHistoryItem])
async def get_history(
    limit: int = Query(default=10, ge=1, le=100, description="Number of queries to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID")
):
    """
    Get query history
    
    Args:
        limit: Maximum number of queries to return (1-100)
        user_id: Optional user ID filter
        
    Returns:
        List of query history items
    """
    try:
        history = get_query_history(limit=limit, user_id=user_id)
        return history
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")


@app.get("/api/stats", response_model=SystemStats)
async def get_stats():
    """
    Get system statistics and analytics
    
    Returns:
        SystemStats with analytics data
    """
    try:
        stats = get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")


@app.delete("/api/history/{query_id}")
async def delete_query(query_id: int):
    """
    Delete a specific query from history
    
    Args:
        query_id: The ID of the query to delete
        
    Returns:
        Success message
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM query_history WHERE id = ?", (query_id,))
        
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Query not found")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted query ID: {query_id}")
        return {"message": f"Query {query_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting query: {str(e)}")


@app.get("/api/rag/stats")
async def get_rag_stats():
    """Get RAG pipeline statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_pipeline.get_stats()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

