# 🚀 Quick Reference Card

## 📁 File Overview

```
📦 Backend Practice Files
│
├── 📘 PRACTICE_SUMMARY.md        ← START HERE! Overview of everything
├── 📕 BACKEND_PRACTICE.md        ← Detailed learning guide
├── 📗 QUICK_REFERENCE.md         ← This file (quick commands)
│
├── ✅ api_server.py              ← STUDY THIS - Complete example
├── ✏️ practice_caching.py        ← IMPLEMENT THIS - Caching task
├── ✏️ practice_auth.py           ← IMPLEMENT THIS - Auth task
│
├── 🧪 test_api_client.py         ← Test your API
└── 🚀 start_api.sh               ← Quick start script
```

---

## ⚡ Quick Start Commands

### 1. Install Dependencies
```bash
pip install fastapi uvicorn
```

### 2. Run the API Server
```bash
# Method 1: Using uvicorn directly
uvicorn api_server:app --reload

# Method 2: Using the start script
./start_api.sh

# API docs available at: http://localhost:8000/docs
```

### 3. Test the API
```bash
# In another terminal window

# Run full test suite
python test_api_client.py

# Interactive mode
python test_api_client.py interactive

# Single query
python test_api_client.py query "What is machine learning?"
```

### 4. Work on Practice Tasks
```bash
# Edit the files
code practice_caching.py    # VS Code
# OR
vim practice_caching.py     # Vim
# OR
nano practice_caching.py    # Nano

# Test your implementation
python practice_caching.py

# Same for auth
code practice_auth.py
python practice_auth.py
```

---

## 🎯 Learning Order

1. **READ** → `PRACTICE_SUMMARY.md` (10 min)
2. **STUDY** → `api_server.py` (30 min)
3. **RUN** → API server and test it (15 min)
4. **READ** → `BACKEND_PRACTICE.md` (20 min)
5. **IMPLEMENT** → `practice_caching.py` (2-3 hours)
6. **IMPLEMENT** → `practice_auth.py` (3-4 hours)
7. **INTEGRATE** → Combine everything (1-2 hours)

**Total Time: ~8-12 hours of focused practice**

---

## 📚 What Each File Teaches

| File | What You Learn | Difficulty |
|------|---------------|------------|
| `api_server.py` | REST API, FastAPI, Database, Error Handling | ✅ Example |
| `practice_caching.py` | LRU Cache, TTL, OrderedDict, Performance | ⭐⭐⭐ |
| `practice_auth.py` | Security, Hashing, Rate Limiting, API Keys | ⭐⭐⭐⭐ |

---

## 🔑 Key Concepts Cheat Sheet

### FastAPI Route
```python
@app.post("/api/endpoint")
async def handler(request: RequestModel):
    # Process
    return ResponseModel(...)
```

### Database Query
```python
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT * FROM table WHERE id = ?", (id,))
result = cursor.fetchone()
conn.close()
```

### Pydantic Model
```python
class MyModel(BaseModel):
    field: str = Field(..., description="...")
    optional_field: Optional[int] = None
```

### Error Handling
```python
if error:
    raise HTTPException(status_code=400, detail="Error message")
```

### Hashing (Security)
```python
import hashlib
hash_value = hashlib.sha256(data.encode()).hexdigest()
```

### Secure Random
```python
import secrets
api_key = f"rag_{secrets.token_hex(16)}"
```

### LRU Cache
```python
from collections import OrderedDict
cache = OrderedDict()
cache[key] = value
cache.move_to_end(key)  # Mark as recently used
cache.popitem(last=False)  # Remove oldest
```

---

## 🧪 Test Commands

### API Endpoints to Test

```bash
# Health check
curl http://localhost:8000/health

# Query the system
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ML?", "user_id": "test"}'

# Get history
curl http://localhost:8000/api/history?limit=5

# Get statistics
curl http://localhost:8000/api/stats

# RAG stats
curl http://localhost:8000/api/rag/stats
```

---

## 🐛 Debugging Tips

### Print Debugging
```python
print(f"Debug: variable = {variable}")
print(f"Type: {type(variable)}")
print(f"Length: {len(variable)}")
```

### Check Database
```bash
sqlite3 data/query_history.db
> SELECT * FROM query_history LIMIT 5;
> .exit
```

### Check if Server is Running
```bash
curl http://localhost:8000/health
```

### View Logs
The server prints logs to console. Look for:
- `INFO` messages (normal operation)
- `WARNING` messages (potential issues)
- `ERROR` messages (problems to fix)

---

## 📊 Progress Checklist

### Day 1
- [ ] Install dependencies
- [ ] Run and test `api_server.py`
- [ ] Read all documentation
- [ ] Understand the complete example

### Day 2
- [ ] Start `practice_caching.py`
- [ ] Implement initialization and key generation
- [ ] Implement get and set methods
- [ ] Test basic caching

### Day 3
- [ ] Complete caching implementation
- [ ] Pass all cache tests
- [ ] Start `practice_auth.py`
- [ ] Implement database and key generation

### Day 4
- [ ] Complete auth implementation
- [ ] Pass all auth tests
- [ ] Integrate into API server
- [ ] Add your own features

---

## 🎓 Success Metrics

**You're making progress when:**
- ✅ You can explain what each endpoint does
- ✅ You understand how FastAPI routing works
- ✅ You can write database queries
- ✅ You understand caching strategies
- ✅ You know security best practices

**You've mastered it when:**
- 🏆 All tests pass
- 🏆 You can add new endpoints yourself
- 🏆 You can explain your code to others
- 🏆 You can integrate everything together
- 🏆 You can deploy to production

---

## 🔗 Quick Links

- **API Docs (when server running):** http://localhost:8000/docs
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **SQLite Docs:** https://www.sqlite.org/docs.html
- **Python Docs:** https://docs.python.org/3/

---

## 💡 Common Issues & Solutions

### "Module not found: fastapi"
```bash
pip install fastapi uvicorn
```

### "Address already in use"
```bash
# Kill the process using port 8000
lsof -ti:8000 | xargs kill -9
# Or use a different port
uvicorn api_server:app --port 8001
```

### "RAG not initialized"
```bash
# Make sure documents exist
ls documents/*.txt
# The server will auto-initialize on startup
```

### Tests failing
```python
# Add print statements to debug
print(f"Debug: {variable}")
# Check if you implemented all TODOs
# Read the error message carefully
```

---

## 🎯 Today's Goal

Pick ONE thing to focus on:
- [ ] Run and understand the API server
- [ ] Implement caching (one method)
- [ ] Implement auth (one method)
- [ ] Test and debug your code
- [ ] Add a new feature

**Small progress is still progress! 🎉**

---

## 📞 When Stuck

1. Read the TODO comments again
2. Look at `api_server.py` for examples
3. Check the test cases
4. Use print() to debug
5. Search Python docs
6. Take a break and come back fresh

---

## 🎉 Celebrate Wins!

- ✅ Ran the server? Great!
- ✅ First test passed? Awesome!
- ✅ Understood a concept? Excellent!
- ✅ Fixed a bug? You're a developer!

**Every line of code you write is progress!**

---

*Good luck with your practice! 💪🚀*

