"""
PRACTICE TASK #1: Implement a Query Caching Layer

OBJECTIVE:
Build a caching system to improve performance by storing and reusing query results.
This reduces repeated calls to the expensive embedding/retrieval pipeline.

LEARNING GOALS:
- Implement caching with TTL (Time To Live)
- Use Redis or in-memory cache
- Handle cache invalidation
- Measure cache hit rates
- Implement cache warming strategies

DIFFICULTY: Intermediate

ESTIMATED TIME: 2-3 hours

INSTRUCTIONS:
Fill in the methods marked with TODO comments. Read the detailed comments to understand
what each method should do, then implement the logic yourself.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass
from collections import OrderedDict
import time


@dataclass
class CacheEntry:
    """
    Represents a single cache entry with its metadata
    
    Attributes:
        key: Unique identifier for the cache entry (hash of the query)
        value: The cached data (answer and sources)
        created_at: Timestamp when entry was created
        expires_at: Timestamp when entry should expire
        hit_count: Number of times this entry has been accessed
    """
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0


class QueryCache:
    """
    LRU (Least Recently Used) Cache with TTL for RAG queries
    
    WHAT YOU NEED TO IMPLEMENT:
    1. Cache storage using OrderedDict (maintains insertion order)
    2. Hash generation for query keys
    3. Cache hit/miss logic
    4. LRU eviction when cache is full
    5. TTL expiration checking
    6. Cache statistics tracking
    
    BONUS CHALLENGES:
    - Add cache persistence to disk
    - Implement cache warming on startup
    - Add cache versioning for when documents change
    - Implement distributed caching with Redis
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the cache
        
        Args:
            max_size: Maximum number of entries to store (default: 100)
            ttl_seconds: Time to live for cache entries in seconds (default: 1 hour)
        
        TODO: Initialize the following instance variables:
        - self.max_size: Store the maximum cache size
        - self.ttl_seconds: Store the TTL duration
        - self.cache: OrderedDict to store cache entries (key -> CacheEntry)
        - self.stats: Dictionary to track hits, misses, evictions
        """
        # TODO: Implement initialization
        # HINT: Use OrderedDict from collections for LRU functionality
        # HINT: Initialize stats with: hits=0, misses=0, evictions=0
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
       
    
    def _generate_cache_key(self, query: str, **kwargs) -> str:
        """
        Generate a unique cache key from query and parameters
        
        WHY THIS IS IMPORTANT:
        Different queries should have different keys. Same query with same parameters
        should produce the same key for cache lookup.
        
        Args:
            query: The user's question
            **kwargs: Additional parameters (user_id, top_k, etc.)
        
        Returns:
            A unique hash string representing this query
        
        TODO: Implement cache key generation
        STEPS:
        1. Create a dictionary with query and all kwargs
        2. Convert dictionary to JSON string (sorted keys for consistency)
        3. Encode the JSON string to bytes
        4. Use hashlib.sha256() to generate hash
        5. Return the hexadecimal digest
        
        EXAMPLE:
        query = "What is machine learning?"
        key = hashlib.sha256(json.dumps({"query": query}, sort_keys=True).encode()).hexdigest()
        """
        # TODO: Implement key generation
        # HINT: json.dumps(data, sort_keys=True) ensures consistent ordering
        # HINT: hashlib.sha256(string.encode()).hexdigest() creates the hash
        data = {"query": query, **kwargs}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()    
        
    
    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result if it exists and is not expired
        
        LOGIC FLOW:
        1. Generate cache key from query
        2. Check if key exists in cache
        3. If exists, check if entry is expired
        4. If expired, remove it and return None (cache miss)
        5. If valid, update hit_count, move to end (LRU), increment hits, return value
        6. If doesn't exist, increment misses, return None
        
        Args:
            query: The user's question
            **kwargs: Additional parameters
        
        Returns:
            Cached value if found and valid, None otherwise
        
        TODO: Implement cache retrieval logic
        STEPS:
        1. Generate cache key using _generate_cache_key()
        2. Check if key is in self.cache
        3. If found:
           a. Get the cache entry
           b. Check if datetime.now() > entry.expires_at (expired?)
           c. If expired: delete from cache, increment stats['misses'], return None
           d. If valid: 
              - Increment entry.hit_count
              - Move entry to end: self.cache.move_to_end(key)
              - Increment self.stats['hits']
              - Return entry.value
        4. If not found: increment self.stats['misses'], return None
        """
        # TODO: Implement get logic
        key = self._generate_cache_key(query, **kwargs)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() > entry.expires_at:
                del self.cache[key]
                self.stats['misses'] += 1
                return None
            
            entry.hit_count += 1
            self.cache.move_to_end(key) #LRU
            self.stats['hits'] += 1
            return entry.value
        
        self.stats['misses'] += 1
        return None
    
    
    def set(self, query: str, value: Dict[str, Any], **kwargs) -> None:
        """
        Store a query result in the cache
        
        LOGIC FLOW:
        1. Generate cache key
        2. Check if cache is full
        3. If full, remove least recently used item (first item in OrderedDict)
        4. Create new CacheEntry with expiration time
        5. Store in cache
        
        Args:
            query: The user's question
            value: The result to cache (answer and sources)
            **kwargs: Additional parameters
        
        TODO: Implement cache storage logic
        STEPS:
        1. Generate cache key using _generate_cache_key()
        2. Check if len(self.cache) >= self.max_size:
           a. If true, evict oldest entry: self.cache.popitem(last=False)
           b. Increment self.stats['evictions']
        3. Create a CacheEntry:
           - key: the generated key
           - value: the provided value
           - created_at: datetime.now()
           - expires_at: datetime.now() + timedelta(seconds=self.ttl_seconds)
           - hit_count: 0
        4. Store entry in self.cache[key] = entry
        """
        # TODO: Implement set logic
        key = self._generate_cache_key(query, **kwargs)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
            
        entry = CacheEntry(key, value, datetime.now(), datetime.now() + timedelta(seconds=self.ttl_seconds), 0)
        self.cache[key] = entry
        # HINT: OrderedDict.popitem(last=False) removes the oldest item (FIFO)
        # HINT: timedelta allows date arithmetic: datetime.now() + timedelta(seconds=60)
        
    
    def invalidate(self, query: Optional[str] = None) -> None:
        """
        Invalidate cache entries
        
        USE CASES:
        - When documents are updated, clear all caches
        - When a specific query needs to be re-computed
        
        Args:
            query: Specific query to invalidate, or None to clear all
        
        TODO: Implement cache invalidation
        STEPS:
        1. If query is None:
           - Clear the entire cache: self.cache.clear()
        2. If query is provided:
           - Generate cache key
           - If key exists in cache, delete it: del self.cache[key]
        """
        # TODO: Implement invalidation logic
        if query is None:
            self.cache.clear()
        else:
            key = self._generate_cache_key(query)
            if key in self.cache:
                del self.cache[key]
       
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring
        
        METRICS TO CALCULATE:
        - Total hits and misses
        - Hit rate (percentage)
        - Current cache size
        - Most frequently accessed queries
        
        Returns:
            Dictionary with cache statistics
        
        TODO: Implement statistics calculation
        STEPS:
        1. Calculate total_requests = hits + misses
        2. Calculate hit_rate = (hits / total_requests * 100) if total > 0 else 0
        3. Get current_size = len(self.cache)
        4. Find top 5 most accessed entries:
           a. Sort cache entries by hit_count (descending)
           b. Take top 5
           c. Create list of {query: ..., hits: ...} dicts
        5. Return dictionary with all stats
        """
        # TODO: Implement statistics
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        current_size = len(self.cache)
        # self.cache.keys() - all cache keys - list of strings (hashes)
        # self.cache.values() - all cache values - list of CacheEntry objects
        # self.cache.items() - all cache items - list of (key, CacheEntry) tuples
        top_5 = sorted(self.cache.values(), key= lambda x: x.hit_count, reverse=True)[:5]
        top_5_dicts = [{'query': entry.key, 'hits': entry.hit_count} for entry in top_5]
        return {
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'current_size': current_size,
            'top_5': top_5_dicts,
            'hits': self.stats['hits'],
            'misses': self.stats['misses']
        }
        # HINT: sorted(iterable, key=lambda x: x.attribute, reverse=True)
        pass
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache
        
        WHEN TO USE:
        - Run periodically in background
        - Before getting statistics
        - When memory pressure is high
        
        Returns:
            Number of entries removed
        
        TODO: Implement expired entry cleanup
        STEPS:
        1. Get current time: now = datetime.now()
        2. Create list of keys to remove (can't modify dict while iterating)
        3. Iterate through self.cache.items():
           - If entry.expires_at < now, add key to removal list
        4. Delete all keys in removal list
        5. Return count of removed entries
        """
        # TODO: Implement cleanup
        now = datetime.now()
        keys_to_remove = []
        for key, entry in self.cache.items():
            if entry.expires_at < now:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.cache[key]
        return len(keys_to_remove)
        # HINT: Build list of expired keys first, then delete them
        # HINT: for key in keys_to_remove: del self.cache[key]
        


# ============================================================================
# Integration Example (Already implemented for you)
# ============================================================================

class CachedRAGPipeline:
    """
    Wrapper around RAG pipeline that adds caching
    
    This is already implemented to show you how to use your cache
    """
    
    def __init__(self, rag_pipeline, cache_size: int = 100, ttl_seconds: int = 3600):
        self.rag_pipeline = rag_pipeline
        self.cache = QueryCache(max_size=cache_size, ttl_seconds=ttl_seconds)
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query with caching"""
        # Try to get from cache
        cached_result = self.cache.get(question, **kwargs)
        if cached_result:
            cached_result['from_cache'] = True
            return cached_result
        
        # Cache miss - query the RAG pipeline
        result = self.rag_pipeline.query(question, **kwargs)
        
        # Store in cache
        self.cache.set(question, result, **kwargs)
        result['from_cache'] = False
        
        return result
    
    def invalidate_cache(self):
        """Invalidate all cached queries"""
        self.cache.invalidate()


# ============================================================================
# Testing Code (Run this after implementing the above)
# ============================================================================

def test_cache():
    """
    Test your cache implementation
    
    RUN THIS AFTER IMPLEMENTING THE ABOVE METHODS
    """
    print("Testing Query Cache Implementation...")
    
    # Create cache
    cache = QueryCache(max_size=3, ttl_seconds=2)
    
    # Test 1: Basic set and get
    print("\n1. Testing basic set/get...")
    cache.set("What is ML?", {"answer": "Machine Learning is..."})
    result = cache.get("What is ML?")
    assert result is not None, "Failed: Should retrieve cached value"
    print("✓ Basic caching works")
    
    # Test 2: Cache miss
    print("\n2. Testing cache miss...")
    result = cache.get("Non-existent query")
    assert result is None, "Failed: Should return None for cache miss"
    print("✓ Cache miss works")
    
    # Test 3: LRU eviction
    print("\n3. Testing LRU eviction...")
    cache.set("Query 1", {"answer": "A1"})
    cache.set("Query 2", {"answer": "A2"})
    cache.set("Query 3", {"answer": "A3"})
    cache.set("Query 4", {"answer": "A4"})  # Should evict "What is ML?"
    
    result = cache.get("What is ML?")   # query 1 should also be evicted
    assert result is None, "Failed: LRU should have evicted oldest entry"
    print("✓ LRU eviction works")
    
    # Test 4: TTL expiration
    print("\n4. Testing TTL expiration...")
    cache.set("Expire soon", {"answer": "This will expire"})
    time.sleep(3)  # Wait for expiration
    result = cache.get("Expire soon")
    assert result is None, "Failed: Entry should have expired"
    print("✓ TTL expiration works")
    
    # Test 5: Statistics
    print("\n5. Testing statistics...")
    stats = cache.get_stats()
    print(f"Stats: {stats}")
    assert 'hits' in stats, "Failed: Stats should contain 'hits'"
    assert 'misses' in stats, "Failed: Stats should contain 'misses'"
    print("✓ Statistics work")
    
    print("\n✅ All tests passed! Your cache implementation is working!")


#UNCOMMENT THIS TO TEST YOUR IMPLEMENTATION:
if __name__ == "__main__":
     test_cache()

