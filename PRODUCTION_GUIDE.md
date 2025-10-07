# Production RAG API Guide

## Overview

This production-ready RAG API integrates three powerful systems:

1. **Authentication System** - API key management with rate limiting
2. **Caching System** - Query result caching for improved performance  
3. **RAG Pipeline** - Document retrieval and question answering

## Features

✅ **API Key Authentication**
- Secure key generation and hashing
- Per-key rate limiting
- Usage tracking and analytics
- Key expiration and revocation

✅ **Query Caching**
- LRU cache with TTL expiration
- Significant performance improvements
- Cache hit rate monitoring
- Easy cache invalidation

✅ **Production Ready**
- Comprehensive error handling
- Request logging and monitoring
- CORS support
- Auto-generated API documentation

## Quick Start

### 1. Start the Server

```bash
# Option 1: Using uvicorn directly
uvicorn api_server_production:app --reload --port 8000

# Option 2: Run as Python script
python api_server_production.py
```

The server will start on `http://localhost:8000`

### 2. View API Documentation

Open your browser to:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Create an API Key

```bash
curl -X POST "http://localhost:8000/api/keys/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My App Key",
    "user_id": "user123",
    "rate_limit": 60,
    "expires_in_days": 365
  }'
```

**Response:**
```json
{
  "api_key": "rag_a1b2c3d4e5f6...",
  "key_id": 1,
  "key_prefix": "rag_a1b2",
  "expires_at": "2026-10-07T08:00:00",
  "message": "⚠️ IMPORTANT: Save this API key now! It won't be shown again."
}
```

⚠️ **Save the API key immediately!** It's only shown once.

### 4. Make Authenticated Requests

```bash
export API_KEY="rag_your_key_here"

curl -X POST "http://localhost:8000/api/query" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "use_cache": true,
    "return_sources": true
  }'
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "sources": ["Source 1...", "Source 2..."],
  "query_id": 42,
  "timestamp": "2025-10-07T08:00:00",
  "from_cache": false,
  "response_time_ms": 245.67
}
```

## API Endpoints

### Public Endpoints (No Auth Required)

#### GET `/` - Root
Get API information and available endpoints.

#### GET `/health` - Health Check
Check system status and cache statistics.

```bash
curl http://localhost:8000/health
```

#### POST `/api/keys/create` - Create API Key
Create a new API key.

```bash
curl -X POST http://localhost:8000/api/keys/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Production Key",
    "user_id": "prod_user",
    "rate_limit": 100,
    "expires_in_days": 365
  }'
```

**Note:** In production, this should be protected with admin authentication.

### Protected Endpoints (Require API Key)

All protected endpoints require the `X-API-Key` header.

#### POST `/api/query` - Query RAG System
Query the RAG system with caching support.

```bash
curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is deep learning?",
    "use_cache": true,
    "return_sources": true
  }'
```

#### GET `/api/cache/stats` - Cache Statistics
Get cache performance metrics.

```bash
curl http://localhost:8000/api/cache/stats \
  -H "X-API-Key: $API_KEY"
```

**Response:**
```json
{
  "total_requests": 150,
  "hits": 95,
  "misses": 55,
  "hit_rate": 63.33,
  "current_size": 45,
  "max_size": 1000,
  "evictions": 2
}
```

#### POST `/api/cache/invalidate` - Clear Cache
Invalidate cache entries.

```bash
# Clear all cache
curl -X POST http://localhost:8000/api/cache/invalidate \
  -H "X-API-Key: $API_KEY"

# Clear specific query
curl -X POST "http://localhost:8000/api/cache/invalidate?query=What+is+ML" \
  -H "X-API-Key: $API_KEY"
```

#### GET `/api/keys/info` - API Key Info
Get information about your API key.

```bash
curl http://localhost:8000/api/keys/info \
  -H "X-API-Key: $API_KEY"
```

#### GET `/api/keys/usage` - Usage Statistics
Get usage statistics for your API key.

```bash
curl http://localhost:8000/api/keys/usage \
  -H "X-API-Key: $API_KEY"
```

**Response:**
```json
{
  "total_requests": 234,
  "successful_requests": 230,
  "success_rate": 98.29,
  "requests_by_endpoint": [
    ["/api/query", 230],
    ["/api/cache/stats", 4]
  ],
  "recent_activity": 45
}
```

#### POST `/api/keys/revoke/{key_id}` - Revoke Key
Revoke an API key.

```bash
curl -X POST http://localhost:8000/api/keys/revoke/1 \
  -H "X-API-Key: $API_KEY"
```

#### GET `/api/history` - Query History
Get your query history.

```bash
curl "http://localhost:8000/api/history?limit=10" \
  -H "X-API-Key: $API_KEY"
```

## Python Client Usage

Use the provided test client:

```python
from test_production_api import ProductionAPIClient

# Initialize client
client = ProductionAPIClient("http://localhost:8000")

# Create API key
client.create_api_key("My App", "user123", rate_limit=60)

# Query the system
result = client.query("What is machine learning?", use_cache=True)
print(f"Answer: {result['answer']}")
print(f"From cache: {result['from_cache']}")
print(f"Time: {result['response_time_ms']}ms")

# Get statistics
cache_stats = client.get_cache_stats()
usage_stats = client.get_usage_stats()

# Get history
history = client.get_history(limit=10)
```

## Performance Benefits

### Cache Performance

The caching system provides significant performance improvements:

```
First Query (Cache Miss):  ~200-500ms
Cached Query (Cache Hit):  ~10-50ms

Speedup: 5-20x faster! 🚀
```

### Real-World Example

```bash
# First query - cache miss
time curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "use_cache": true}'

# Response: "from_cache": false, "response_time_ms": 342.56

# Second query - cache hit
time curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "use_cache": true}'

# Response: "from_cache": true, "response_time_ms": 23.45
```

## Rate Limiting

Each API key has a configurable rate limit (default: 60 requests/minute).

When rate limit is exceeded:
```json
{
  "detail": "Rate limit exceeded"
}
```

**Status Code:** 401 Unauthorized

## Security Best Practices

### 1. API Key Management

✅ **DO:**
- Store API keys in environment variables
- Use different keys for dev/staging/production
- Rotate keys periodically
- Set appropriate rate limits
- Use key expiration

❌ **DON'T:**
- Commit API keys to version control
- Share keys between users/applications
- Use the same key for all environments
- Store keys in plain text in config files

### 2. Production Deployment

```python
# In production, protect key creation endpoint
@app.post("/api/keys/create")
async def create_api_key(
    request: CreateAPIKeyRequest,
    admin_key: str = Depends(validate_admin_key)  # Add admin auth
):
    # ... key creation logic
```

### 3. CORS Configuration

Update CORS settings for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
```

## Monitoring & Analytics

### Cache Hit Rate

Monitor cache hit rate to optimize performance:

```python
cache_stats = client.get_cache_stats()
hit_rate = cache_stats['hit_rate']

if hit_rate < 40:
    print("⚠️ Low cache hit rate - consider increasing TTL")
elif hit_rate > 80:
    print("✅ Excellent cache performance!")
```

### Usage Tracking

Track API usage per key:

```python
usage = client.get_usage_stats()
print(f"Success rate: {usage['success_rate']}%")
print(f"Recent activity: {usage['recent_activity']} queries in 24h")
```

## Troubleshooting

### Problem: "Authentication system not initialized"

**Solution:** Ensure the server started properly. Check logs for initialization errors.

### Problem: "Rate limit exceeded"

**Solution:** Wait 60 seconds or create a key with higher rate limit.

### Problem: "RAG system not initialized"

**Solution:** Add documents to the `./documents/` folder and restart the server.

### Problem: Cache not working

**Solution:** 
1. Check cache stats with `/api/cache/stats`
2. Ensure `use_cache: true` in query request
3. Verify TTL hasn't expired (default: 1 hour)

## Configuration

### Cache Settings

Adjust cache settings in `startup_event()`:

```python
query_cache = QueryCache(
    max_size=1000,      # Maximum entries
    ttl_seconds=3600    # 1 hour TTL
)
```

### RAG Configuration

Adjust RAG settings:

```python
config = RAGConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k_retrieval=5,
    use_hybrid_search=True,   # Enable for better results
    use_reranking=True,       # Requires Cohere API key
)
```

## Testing

Run the complete demo:

```bash
# Start server in one terminal
uvicorn api_server_production:app --reload

# Run tests in another terminal
python test_production_api.py
```

The demo will:
1. ✅ Create an API key
2. ✅ Test authentication
3. ✅ Demonstrate caching performance
4. ✅ Show rate limiting
5. ✅ Display monitoring stats

## Next Steps

1. **Add Documents:** Place documents in `./documents/` folder
2. **Configure Environment:** Set OpenAI/Cohere API keys in `.env`
3. **Customize Rate Limits:** Adjust per user/application needs
4. **Add Admin Auth:** Protect key creation endpoint
5. **Deploy to Production:** Use proper HTTPS and domain
6. **Monitor Performance:** Track cache hit rates and usage

## Support

For issues or questions:
- Check API docs: http://localhost:8000/docs
- Review logs for detailed error messages
- Test with the provided client: `test_production_api.py`

---

**Built with:** FastAPI, SQLite, Python 3.9+

**License:** MIT

