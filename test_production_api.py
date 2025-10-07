"""
Test client for the production RAG API with authentication and caching

This script demonstrates:
1. Creating an API key
2. Making authenticated requests
3. Testing cache performance
4. Monitoring API key usage
5. Cache statistics
"""

import requests
import time
from typing import Dict, Any

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = None  # Will be set after creation


class ProductionAPIClient:
    """Client for interacting with the production RAG API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.api_key = None
    
    def create_api_key(self, name: str, user_id: str, rate_limit: int = 60) -> Dict[str, Any]:
        """Create a new API key"""
        print(f"\n{'='*60}")
        print("Creating API Key...")
        print(f"{'='*60}")
        
        response = requests.post(
            f"{self.base_url}/api/keys/create",
            json={
                "name": name,
                "user_id": user_id,
                "rate_limit": rate_limit
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.api_key = data['api_key']
            print(f"✅ API Key Created Successfully!")
            print(f"   Key ID: {data['key_id']}")
            print(f"   Key Prefix: {data['key_prefix']}")
            print(f"   Full Key: {data['api_key']}")
            print(f"   ⚠️  {data['message']}")
            return data
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to create API key: {response.text}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        print(f"\n{'='*60}")
        print("System Health Check")
        print(f"{'='*60}")
        
        response = requests.get(f"{self.base_url}/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"RAG Initialized: {data['rag_initialized']}")
        print(f"Auth Initialized: {data['auth_initialized']}")
        print(f"Cache Initialized: {data['cache_initialized']}")
        
        if data.get('cache_stats'):
            stats = data['cache_stats']
            print(f"\nCache Statistics:")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Hits: {stats['hits']}")
            print(f"  Misses: {stats['misses']}")
            print(f"  Hit Rate: {stats['hit_rate']}%")
            print(f"  Current Size: {stats['current_size']}/{stats['max_size']}")
        
        return data
    
    def query(self, question: str, use_cache: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.api_key:
            raise Exception("No API key set. Create one first.")
        
        headers = {"X-API-Key": self.api_key}
        
        response = requests.post(
            f"{self.base_url}/api/query",
            json={
                "question": question,
                "use_cache": use_cache,
                "return_sources": True
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print(f"❌ Authentication Error: {response.json()['detail']}")
            raise Exception("Authentication failed")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            raise Exception(f"Query failed: {response.text}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.api_key:
            raise Exception("No API key set. Create one first.")
        
        headers = {"X-API-Key": self.api_key}
        response = requests.get(f"{self.base_url}/api/cache/stats", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get cache stats: {response.text}")
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get current API key information"""
        if not self.api_key:
            raise Exception("No API key set. Create one first.")
        
        headers = {"X-API-Key": self.api_key}
        response = requests.get(f"{self.base_url}/api/keys/info", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get key info: {response.text}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API key usage statistics"""
        if not self.api_key:
            raise Exception("No API key set. Create one first.")
        
        headers = {"X-API-Key": self.api_key}
        response = requests.get(f"{self.base_url}/api/keys/usage", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get usage stats: {response.text}")
    
    def get_history(self, limit: int = 10) -> list:
        """Get query history"""
        if not self.api_key:
            raise Exception("No API key set. Create one first.")
        
        headers = {"X-API-Key": self.api_key}
        response = requests.get(
            f"{self.base_url}/api/history",
            params={"limit": limit},
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get history: {response.text}")


def demo_authentication():
    """Demonstrate authentication features"""
    client = ProductionAPIClient()
    
    # Create API key
    client.create_api_key(
        name="Demo Key",
        user_id="demo_user",
        rate_limit=60
    )
    
    # Get key info
    print(f"\n{'='*60}")
    print("API Key Information")
    print(f"{'='*60}")
    key_info = client.get_key_info()
    print(f"Name: {key_info['name']}")
    print(f"User ID: {key_info['user_id']}")
    print(f"Rate Limit: {key_info['rate_limit']} requests/minute")
    print(f"Usage Count: {key_info['usage_count']}")
    
    return client


def demo_caching(client: ProductionAPIClient):
    """Demonstrate caching performance"""
    print(f"\n{'='*60}")
    print("Testing Cache Performance")
    print(f"{'='*60}")
    
    test_question = "What is machine learning?"
    
    # First query (cache miss)
    print(f"\n1️⃣ First Query (should be CACHE MISS)...")
    start = time.time()
    result1 = client.query(test_question, use_cache=True)
    time1 = (time.time() - start) * 1000
    
    print(f"   Answer: {result1['answer'][:100]}...")
    print(f"   From Cache: {result1['from_cache']}")
    print(f"   Response Time: {time1:.2f}ms")
    
    # Second query (cache hit)
    print(f"\n2️⃣ Second Query (should be CACHE HIT)...")
    start = time.time()
    result2 = client.query(test_question, use_cache=True)
    time2 = (time.time() - start) * 1000
    
    print(f"   Answer: {result2['answer'][:100]}...")
    print(f"   From Cache: {result2['from_cache']}")
    print(f"   Response Time: {time2:.2f}ms")
    
    # Performance comparison
    if time1 > 0:
        speedup = time1 / time2
        print(f"\n🚀 Cache Performance:")
        print(f"   Cache Miss: {time1:.2f}ms")
        print(f"   Cache Hit: {time2:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x faster!")


def demo_rate_limiting(client: ProductionAPIClient):
    """Demonstrate rate limiting"""
    print(f"\n{'='*60}")
    print("Testing Rate Limiting")
    print(f"{'='*60}")
    
    # Make multiple rapid requests
    print("\nMaking 5 rapid requests...")
    for i in range(5):
        try:
            result = client.query(f"Test query {i}", use_cache=False)
            print(f"  Request {i+1}: ✅ Success (time: {result['response_time_ms']:.2f}ms)")
        except Exception as e:
            print(f"  Request {i+1}: ❌ Failed - {e}")


def demo_monitoring(client: ProductionAPIClient):
    """Demonstrate monitoring capabilities"""
    print(f"\n{'='*60}")
    print("Monitoring & Analytics")
    print(f"{'='*60}")
    
    # Cache statistics
    print("\n📊 Cache Statistics:")
    cache_stats = client.get_cache_stats()
    print(f"   Total Requests: {cache_stats['total_requests']}")
    print(f"   Cache Hits: {cache_stats['hits']}")
    print(f"   Cache Misses: {cache_stats['misses']}")
    print(f"   Hit Rate: {cache_stats['hit_rate']}%")
    print(f"   Current Size: {cache_stats['current_size']}/{cache_stats['max_size']}")
    
    # API key usage
    print("\n📈 API Key Usage:")
    usage = client.get_usage_stats()
    print(f"   Total Requests: {usage['total_requests']}")
    print(f"   Successful: {usage['successful_requests']}")
    print(f"   Success Rate: {usage['success_rate']}%")
    print(f"   Recent Activity (24h): {usage['recent_activity']}")
    
    # Query history
    print("\n📜 Recent Queries:")
    history = client.get_history(limit=5)
    for i, item in enumerate(history, 1):
        print(f"   {i}. {item['question'][:50]}...")
        print(f"      Cache: {item['from_cache']}, Time: {item['response_time_ms']:.2f}ms")


def run_complete_demo():
    """Run the complete demonstration"""
    print(f"\n{'#'*60}")
    print("# Production RAG API - Complete Feature Demo")
    print(f"{'#'*60}")
    
    try:
        # Check if server is running
        try:
            requests.get(BASE_URL)
        except:
            print("\n❌ Error: Server is not running!")
            print("   Start the server with: uvicorn api_server_production:app --reload")
            return
        
        # Run demos
        client = demo_authentication()
        time.sleep(1)
        
        demo_caching(client)
        time.sleep(1)
        
        demo_rate_limiting(client)
        time.sleep(1)
        
        demo_monitoring(client)
        
        # Health check
        client.health_check()
        
        print(f"\n{'#'*60}")
        print("# ✅ Demo completed successfully!")
        print(f"{'#'*60}")
        
        print(f"\n💡 Pro Tips:")
        print("   - Store your API key securely (environment variable)")
        print("   - Monitor cache hit rates to optimize performance")
        print("   - Check rate limits to avoid throttling")
        print("   - Review query history for insights")
        
        print(f"\n📚 API Documentation: {BASE_URL}/docs")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_complete_demo()

