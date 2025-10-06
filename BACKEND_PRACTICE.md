# 🐍 Backend Python Development Practice Guide

Welcome! This guide provides hands-on practice for backend Python development using the RAG project.

## 📋 Table of Contents

1. [Complete Example: REST API with Query History](#complete-example)
2. [Practice Task #1: Query Caching Layer](#practice-1)
3. [Practice Task #2: API Authentication](#practice-2)
4. [Setup Instructions](#setup)
5. [Learning Resources](#resources)

---

## 🎯 Complete Example: REST API with Query History {#complete-example}

**File:** `api_server.py`

This is a **fully implemented** REST API that demonstrates professional backend development practices.

### What You'll Learn

- ✅ RESTful API design with FastAPI
- ✅ Database integration with SQLite
- ✅ Request/Response validation with Pydantic
- ✅ Error handling and logging
- ✅ CORS configuration
- ✅ API documentation (auto-generated)
- ✅ Query history tracking
- ✅ Analytics and statistics

### How to Run

```bash
# Install FastAPI and dependencies
pip install fastapi uvicorn

# Start the server
uvicorn api_server:app --reload

# Open API documentation
# Visit: http://localhost:8000/docs
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/query` | POST | Query the RAG system |
| `/api/history` | GET | Get query history |
| `/api/stats` | GET | Get system statistics |
| `/api/history/{id}` | DELETE | Delete a query |
| `/api/rag/stats` | GET | Get RAG pipeline stats |

### Test the API

```bash
# Query the RAG system
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "user_id": "user123"}'

# Get query history
curl "http://localhost:8000/api/history?limit=10"

# Get statistics
curl "http://localhost:8000/api/stats"
```

### Key Concepts Demonstrated

1. **FastAPI Application Structure**
   ```python
   app = FastAPI(title="...", description="...")
   
   @app.post("/api/query")
   async def query_rag(request: QueryRequest):
       # Handler logic
   ```

2. **Pydantic Models for Validation**
   ```python
   class QueryRequest(BaseModel):
       question: str = Field(..., min_length=1)
       user_id: Optional[str] = None
   ```

3. **Database Operations**
   ```python
   conn = sqlite3.connect(DB_PATH)
   cursor = conn.cursor()
   cursor.execute("INSERT INTO ...")
   conn.commit()
   ```

4. **Error Handling**
   ```python
   if not rag_pipeline:
       raise HTTPException(status_code=503, detail="...")
   ```

---

## 🏋️ Practice Task #1: Query Caching Layer {#practice-1}

**File:** `practice_caching.py`

**Difficulty:** ⭐⭐⭐ Intermediate

**Estimated Time:** 2-3 hours

### Objective

Build a caching system to improve API performance by storing and reusing query results.

### What You'll Learn

- LRU (Least Recently Used) cache implementation
- TTL (Time To Live) expiration
- Cache key generation with hashing
- Cache statistics tracking
- Memory-efficient data structures

### Tasks to Implement

1. ✏️ **Cache Initialization** (`__init__`)
   - Set up OrderedDict for LRU behavior
   - Initialize statistics tracking

2. ✏️ **Key Generation** (`_generate_cache_key`)
   - Create unique hashes for queries
   - Handle additional parameters

3. ✏️ **Cache Retrieval** (`get`)
   - Implement cache hit/miss logic
   - Check TTL expiration
   - Update LRU order

4. ✏️ **Cache Storage** (`set`)
   - Handle cache size limits
   - Implement LRU eviction
   - Set expiration times

5. ✏️ **Cache Invalidation** (`invalidate`)
   - Clear specific or all entries
   - Handle document updates

6. ✏️ **Statistics** (`get_stats`)
   - Calculate hit rate
   - Track popular queries

7. ✏️ **Cleanup** (`cleanup_expired`)
   - Remove expired entries

### How to Practice

1. Open `practice_caching.py`
2. Read the detailed comments for each method
3. Implement the logic marked with `TODO`
4. Test with: `python practice_caching.py`

### Success Criteria

All tests pass when you run:
```python
if __name__ == "__main__":
    test_cache()
```

### Hints

- Use `OrderedDict` from `collections` for LRU
- Use `hashlib.sha256()` for secure hashing
- Use `datetime` and `timedelta` for expiration
- Move accessed items to end: `dict.move_to_end(key)`
- Remove oldest: `dict.popitem(last=False)`

---

## 🔐 Practice Task #2: API Authentication System {#practice-2}

**File:** `practice_auth.py`

**Difficulty:** ⭐⭐⭐⭐ Intermediate-Advanced

**Estimated Time:** 3-4 hours

### Objective

Build a secure API key authentication system to protect your endpoints and track usage.

### What You'll Learn

- Secure key generation with `secrets` module
- Password hashing (never store raw keys!)
- Rate limiting implementation
- Usage tracking and analytics
- Security best practices

### Tasks to Implement

1. ✏️ **Database Setup** (`_init_database`)
   - Create tables for keys and usage
   - Set up proper indexes

2. ✏️ **Key Generation** (`generate_api_key`)
   - Generate cryptographically secure keys
   - Hash keys before storage
   - Handle expiration

3. ✏️ **Key Validation** (`validate_api_key`)
   - Check key format
   - Verify against database
   - Check expiration and rate limits

4. ✏️ **Rate Limiting** (`_check_rate_limit`)
   - Count requests in time window
   - Reject if limit exceeded

5. ✏️ **Usage Recording** (`record_usage`)
   - Track API calls
   - Update statistics

6. ✏️ **Key Management** (`revoke_api_key`, `get_user_keys`)
   - Revoke compromised keys
   - List user's keys

7. ✏️ **Analytics** (`get_usage_stats`)
   - Calculate usage metrics
   - Track endpoint popularity

### Security Best Practices

⚠️ **NEVER** store raw API keys in the database!
- ✅ Hash with SHA-256: `hashlib.sha256(key.encode()).hexdigest()`
- ✅ Use `secrets` module: `secrets.token_hex(16)`
- ✅ Implement rate limiting
- ✅ Support key rotation
- ✅ Log authentication attempts

### How to Practice

1. Open `practice_auth.py`
2. Read the detailed comments for each method
3. Implement the logic marked with `TODO`
4. Test with: `python practice_auth.py`

### Success Criteria

All tests pass when you run:
```python
if __name__ == "__main__":
    test_api_key_manager()
```

### Integration Example

Once implemented, integrate with FastAPI:

```python
from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from practice_auth import APIKeyManager

api_key_header = APIKeyHeader(name="X-API-Key")
manager = APIKeyManager()

async def validate_key(key: str = Security(api_key_header)):
    validation = manager.validate_api_key(key)
    if not validation.is_valid:
        raise HTTPException(401, detail=validation.error_message)
    return validation.api_key

@app.post("/api/query")
async def query(request: QueryRequest, key = Depends(validate_key)):
    # Protected endpoint
    pass
```

---

## 🚀 Setup Instructions {#setup}

### Prerequisites

```bash
python --version  # Python 3.8+
```

### Install Dependencies

```bash
# Core dependencies (should already be installed)
pip install -r requirements.txt

# Additional dependencies for practice
pip install fastapi uvicorn
```

### Project Structure

```
RAG/
├── api_server.py           # ✅ Complete example (study this)
├── practice_caching.py     # ✏️ Your implementation
├── practice_auth.py        # ✏️ Your implementation
├── src/                    # RAG system source code
│   ├── rag_pipeline.py
│   ├── config.py
│   └── ...
└── data/                   # Generated databases
    ├── query_history.db    # Created by api_server.py
    ├── api_keys.db         # Created by practice_auth.py
    └── vectorstore/        # RAG embeddings
```

---

## 📚 Learning Resources {#resources}

### Backend Concepts

1. **REST API Design**
   - [REST API Tutorial](https://restfulapi.net/)
   - HTTP methods: GET, POST, PUT, DELETE
   - Status codes: 200, 400, 401, 404, 500

2. **FastAPI**
   - [Official Docs](https://fastapi.tiangolo.com/)
   - Automatic validation with Pydantic
   - Interactive API docs

3. **Database Management**
   - [SQLite Tutorial](https://www.sqlitetutorial.net/)
   - SQL basics: SELECT, INSERT, UPDATE, DELETE
   - Indexing for performance

4. **Security**
   - [OWASP Top 10](https://owasp.org/www-project-top-ten/)
   - Hashing vs Encryption
   - Rate limiting strategies

### Python Modules Used

| Module | Purpose | Documentation |
|--------|---------|---------------|
| `fastapi` | Web framework | [docs](https://fastapi.tiangolo.com/) |
| `pydantic` | Data validation | [docs](https://pydantic-docs.helpmanual.io/) |
| `sqlite3` | Database | [docs](https://docs.python.org/3/library/sqlite3.html) |
| `hashlib` | Hashing | [docs](https://docs.python.org/3/library/hashlib.html) |
| `secrets` | Secure random | [docs](https://docs.python.org/3/library/secrets.html) |
| `collections` | OrderedDict | [docs](https://docs.python.org/3/library/collections.html) |

---

## 🎓 Learning Path

### Beginner → Intermediate

1. **Study** `api_server.py` completely
   - Understand the structure
   - Try modifying endpoints
   - Add new features

2. **Implement** `practice_caching.py`
   - Start with simple methods
   - Use the tests to guide you
   - Gradually add complexity

3. **Implement** `practice_auth.py`
   - Build on caching knowledge
   - Focus on security
   - Integrate with API

### Advanced Challenges

Once you complete the practice tasks, try these:

1. **Redis Caching**
   - Replace in-memory cache with Redis
   - Implement distributed caching

2. **JWT Authentication**
   - Replace API keys with JWT tokens
   - Implement token refresh

3. **Background Tasks**
   - Use Celery for async processing
   - Process documents in background

4. **API Rate Limiting**
   - Implement sliding window algorithm
   - Add user-specific quotas

5. **Monitoring & Logging**
   - Add structured logging
   - Implement health metrics
   - Create dashboards

---

## 💡 Tips for Success

1. **Read the Comments Carefully**
   - Each TODO has detailed instructions
   - Hints are provided for tricky parts

2. **Test Frequently**
   - Run tests after each method
   - Use print statements for debugging

3. **Use Type Hints**
   - Helps catch errors early
   - Makes code self-documenting

4. **Don't Skip Error Handling**
   - Always consider edge cases
   - Use try-except blocks

5. **Ask Questions**
   - Google is your friend
   - Check documentation
   - Look at the complete example

---

## 🎯 Next Steps

After completing these exercises:

1. ✅ Integrate caching into `api_server.py`
2. ✅ Add authentication to protect endpoints
3. ✅ Deploy your API (Heroku, Railway, AWS)
4. ✅ Add more advanced features
5. ✅ Build a frontend to consume the API

---

## 🤝 Getting Help

If you're stuck:

1. **Review the complete example** in `api_server.py`
2. **Read the test cases** to understand expected behavior
3. **Check Python documentation** for the modules used
4. **Debug with print statements** to see what's happening
5. **Break problems into smaller pieces**

---

Good luck with your practice! 🚀

Remember: **The best way to learn is by doing!**

