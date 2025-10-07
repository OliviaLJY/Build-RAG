# Production RAG API - System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Application                        │
│                     (curl / Python / Browser)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP Request
                            │ X-API-Key: rag_...
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                           │
│                   (api_server_production.py)                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Authentication Middleware                  │    │
│  │                (validate_api_key_dependency)            │    │
│  │                                                          │    │
│  │  1. Extract X-API-Key header                           │    │
│  │  2. Validate with APIKeyManager                        │    │
│  │  3. Check rate limits                                  │    │
│  │  4. Record usage                                       │    │
│  └──────────────────┬─────────────────────────────────────┘    │
│                     │ Authenticated Request                     │
│                     ▼                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                   Query Handler                         │    │
│  │                  (/api/query endpoint)                  │    │
│  │                                                          │    │
│  │  Step 1: Check QueryCache                              │    │
│  │  ├─ Cache Hit? → Return cached result (20ms) ⚡       │    │
│  │  └─ Cache Miss? → Continue to Step 2                  │    │
│  │                                                          │    │
│  │  Step 2: Query RAGPipeline                             │    │
│  │  ├─ Retrieve relevant documents                        │    │
│  │  ├─ Generate answer with LLM                           │    │
│  │  └─ Return result (300ms)                              │    │
│  │                                                          │    │
│  │  Step 3: Cache result for future use                   │    │
│  │  Step 4: Record to history database                    │    │
│  │  Step 5: Return response to client                     │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   APIKeyManager  │     │   QueryCache     │     │   RAGPipeline    │
│                  │     │                  │     │                  │
│ - Generate keys  │     │ - LRU eviction   │     │ - Load docs      │
│ - Validate keys  │     │ - TTL expiration │     │ - Embed queries  │
│ - Rate limiting  │     │ - Hit/miss stats │     │ - Retrieve docs  │
│ - Usage tracking │     │ - Invalidation   │     │ - Generate answer│
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │   SQLite Databases      │
                     │                         │
                     │ - api_keys.db          │
                     │ - api_key_usage.db     │
                     │ - query_history.db     │
                     │                         │
                     │ - ChromaDB/FAISS       │
                     │   (Vector Store)       │
                     └─────────────────────────┘
```

## Request Flow Diagram

### Scenario 1: First Query (Cache Miss)

```
Client                FastAPI              Auth              Cache             RAG
  │                      │                   │                 │                 │
  ├──POST /api/query────►│                   │                 │                 │
  │  X-API-Key: rag_...  │                   │                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Validate Key────►│                 │                 │
  │                      │◄─────Valid────────┤                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Check Rate Limit►│                 │                 │
  │                      │◄────OK (45/60)────┤                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Get Cached?──────┼────────────────►│                 │
  │                      │◄──Cache Miss──────┼─────────────────┤                 │
  │                      │                   │                 │                 │
  │                      ├──Query────────────┼─────────────────┼────────────────►│
  │                      │                   │                 │  1. Embed query │
  │                      │                   │                 │  2. Search docs │
  │                      │                   │                 │  3. Generate    │
  │                      │◄──Answer──────────┼─────────────────┼─────────────────┤
  │                      │  (300ms)          │                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Cache Result─────┼────────────────►│                 │
  │                      │                   │                 │                 │
  │                      ├──Record Usage────►│                 │                 │
  │                      │                   │                 │                 │
  │◄─────Response────────┤                   │                 │                 │
  │  from_cache: false   │                   │                 │                 │
  │  time: 342ms         │                   │                 │                 │
```

### Scenario 2: Repeated Query (Cache Hit)

```
Client                FastAPI              Auth              Cache             RAG
  │                      │                   │                 │                 │
  ├──POST /api/query────►│                   │                 │                 │
  │  X-API-Key: rag_...  │                   │                 │                 │
  │  (same question)     │                   │                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Validate Key────►│                 │                 │
  │                      │◄─────Valid────────┤                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Check Rate Limit►│                 │                 │
  │                      │◄────OK (46/60)────┤                 │                 │
  │                      │                   │                 │                 │
  │                      ├──Get Cached?──────┼────────────────►│                 │
  │                      │◄──Cache Hit! ⚡───┼─────────────────┤                 │
  │                      │  (instant)        │                 │   (not called)  │
  │                      │                   │                 │                 │
  │                      ├──Record Usage────►│                 │                 │
  │                      │                   │                 │                 │
  │◄─────Response────────┤                   │                 │                 │
  │  from_cache: true    │                   │                 │                 │
  │  time: 23ms ⚡       │                   │                 │                 │
```

## Data Flow

### API Key Creation

```
POST /api/keys/create
        │
        ▼
APIKeyManager.generate_api_key()
        │
        ├─1. Generate random key: rag_{32_hex_chars}
        ├─2. Hash with SHA-256
        ├─3. Store hash in api_keys.db
        └─4. Return raw key (only once!)
```

### Query Processing

```
POST /api/query
        │
        ▼
Authentication
        ├─ Hash provided key
        ├─ Lookup in api_keys.db
        ├─ Check expiration
        └─ Check rate limit
              │
              ▼
        Cache Lookup
              ├─ Generate cache key (hash of query + params)
              ├─ Check if exists in cache
              └─ Check if expired (TTL)
                    │
                    ├─ Cache Hit ─────► Return cached result
                    │                   (15x faster!)
                    │
                    └─ Cache Miss ─────► Query RAG Pipeline
                                              │
                                              ├─ Retrieve documents
                                              ├─ Generate answer
                                              ├─ Cache result
                                              └─ Return result
```

## Database Schema

### api_keys.db

```sql
CREATE TABLE api_keys (
    key_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,  -- SHA-256 hash
    user_id TEXT NOT NULL,
    created_at DATETIME,
    expires_at DATETIME,            -- NULL = never expires
    is_active BOOLEAN,
    rate_limit INTEGER,             -- requests per minute
    usage_count INTEGER
);

CREATE TABLE api_key_usage (
    id INTEGER PRIMARY KEY,
    key_id INTEGER,
    timestamp DATETIME,
    endpoint TEXT,
    success BOOLEAN
);
```

### query_history.db

```sql
CREATE TABLE query_history (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT,                   -- JSON array
    user_id TEXT,
    api_key_id INTEGER,
    timestamp DATETIME,
    response_time_ms REAL,
    from_cache BOOLEAN
);
```

## Cache Architecture

### LRU Cache Structure

```
OrderedDict (max_size=1000, ttl=1h)
┌─────────────────────────────────────┐
│ Key (hash)       │ CacheEntry        │
├──────────────────┼───────────────────┤
│ abc123...        │ - value           │
│                  │ - created_at      │
│                  │ - expires_at      │
│                  │ - hit_count: 45   │
├──────────────────┼───────────────────┤
│ def456...        │ - value           │
│                  │ - created_at      │
│                  │ - expires_at      │
│                  │ - hit_count: 12   │
└──────────────────┴───────────────────┘
     ↑                        ↓
     │    LRU Eviction        │
     │    (when full)         │
     └────────────────────────┘
```

## Security Layers

```
┌────────────────────────────────────────┐
│         Request from Client            │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│     Layer 1: CORS Validation           │
│  - Check origin allowed                │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│     Layer 2: API Key Validation        │
│  - Key format check (rag_...)          │
│  - Hash and lookup in database         │
│  - Check if active                     │
│  - Check expiration                    │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│     Layer 3: Rate Limiting             │
│  - Count requests in last 60s          │
│  - Compare with key's rate_limit       │
│  - Reject if exceeded                  │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│     Layer 4: Input Validation          │
│  - Pydantic model validation           │
│  - Length checks                       │
│  - Type checking                       │
└───────────────┬────────────────────────┘
                │
                ▼
┌────────────────────────────────────────┐
│     Request Processed ✅               │
└────────────────────────────────────────┘
```

## Performance Optimization

### 1. Caching Layer

```
Without Cache:
  Every query → RAG Pipeline → 300ms

With Cache:
  First query → RAG Pipeline → 300ms → Cache result
  Repeated query → Cache → 20ms ⚡

Cache Hit Rate = Hits / (Hits + Misses)
Target: > 60% for optimal performance
```

### 2. Database Indexing

```sql
-- Fast lookups
CREATE INDEX idx_key_hash ON api_keys (key_hash);
CREATE INDEX idx_timestamp ON api_key_usage (timestamp);
CREATE INDEX idx_user_id ON query_history (user_id);
```

### 3. Connection Pooling

```python
# Each request opens/closes connection
# For production, use connection pooling:
# - SQLAlchemy with pool
# - PostgreSQL instead of SQLite
```

## Scaling Considerations

### Current Architecture (Single Server)

```
Client ──► FastAPI Server ──► SQLite
                │
                └──► In-Memory Cache
```

**Handles:** ~100-1000 req/sec (depending on cache hit rate)

### Scaled Architecture (Production)

```
                    ┌──► FastAPI Server 1 ──┐
Clients ──► Load    ├──► FastAPI Server 2 ──┼──► PostgreSQL
            Balancer├──► FastAPI Server 3 ──┤    (Multi-master)
                    └──► FastAPI Server N ──┘
                              │
                              └──► Redis Cluster
                                   (Distributed Cache)
```

**Handles:** 10,000+ req/sec

## Monitoring Points

```
┌─────────────────────────────────────────────┐
│           Monitoring Dashboard              │
├─────────────────────────────────────────────┤
│                                             │
│  Cache Metrics:                             │
│  ├─ Hit Rate: 67.5%                        │
│  ├─ Size: 234/1000                         │
│  └─ Evictions: 12                          │
│                                             │
│  API Key Metrics (per key):                │
│  ├─ Request Count: 1,234                   │
│  ├─ Success Rate: 98.3%                    │
│  ├─ Rate Limit Usage: 45/60 (75%)         │
│  └─ Avg Response Time: 87ms                │
│                                             │
│  System Metrics:                            │
│  ├─ Active Keys: 23                        │
│  ├─ Total Queries Today: 5,432            │
│  ├─ Cache Memory: 45MB                     │
│  └─ Database Size: 234MB                   │
└─────────────────────────────────────────────┘
```

## Error Handling Flow

```
Request
   │
   ▼
┌─────────────────────┐
│ Try: Process Query  │
└──────┬──────────────┘
       │
       ├──► Success ──► Record usage (success=True)
       │                Return result
       │
       └──► Error
              │
              ├──► 401: Auth Error
              │      └─ Invalid/expired/rate limited key
              │
              ├──► 503: Service Error
              │      └─ RAG/Cache not initialized
              │
              └──► 500: Internal Error
                     └─ Record usage (success=False)
                        Log error
                        Return error message
```

---

## Summary

This architecture provides:

✅ **Security** - Multi-layer authentication and validation
✅ **Performance** - 15x speedup with intelligent caching
✅ **Scalability** - Can be scaled horizontally
✅ **Reliability** - Comprehensive error handling
✅ **Observability** - Full monitoring and analytics

**Built for production use!** 🚀

