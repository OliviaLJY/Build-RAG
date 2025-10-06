"""
Test Client for RAG API

This script demonstrates how to interact with the REST API.
Run the API server first: uvicorn api_server:app --reload
"""

import requests
import json
from datetime import datetime


# API base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_health_check():
    """Test the health check endpoint"""
    print_section("1. Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json()


def test_query(question, user_id="test_user"):
    """Test querying the RAG system"""
    print_section("2. Query RAG System")
    
    payload = {
        "question": question,
        "user_id": user_id,
        "session_id": "test_session"
    }
    
    print(f"Question: {question}")
    print(f"User ID: {user_id}")
    print("\nSending request...")
    
    response = requests.post(
        f"{BASE_URL}/api/query",
        json=payload
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery ID: {result['query_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"\nAnswer:\n{result['answer'][:200]}...")
        print(f"\nNumber of sources: {len(result['sources'])}")
        return result
    else:
        print(f"Error: {response.text}")
        return None


def test_query_history(limit=5):
    """Test retrieving query history"""
    print_section("3. Query History")
    
    response = requests.get(
        f"{BASE_URL}/api/history",
        params={"limit": limit}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        history = response.json()
        print(f"\nFound {len(history)} queries in history:")
        
        for i, item in enumerate(history, 1):
            print(f"\n{i}. Query ID: {item['id']}")
            print(f"   Question: {item['question'][:50]}...")
            print(f"   User: {item['user_id']}")
            print(f"   Response Time: {item['response_time_ms']:.2f}ms")
            print(f"   Timestamp: {item['timestamp']}")
        
        return history
    else:
        print(f"Error: {response.text}")
        return None


def test_statistics():
    """Test getting system statistics"""
    print_section("4. System Statistics")
    
    response = requests.get(f"{BASE_URL}/api/stats")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print("\nSystem Statistics:")
        print(f"  Total Queries: {stats['total_queries']}")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")
        
        if stats['most_common_topics']:
            print("\n  Most Common Questions:")
            for topic in stats['most_common_topics'][:3]:
                print(f"    - {topic['question'][:40]}... ({topic['count']} times)")
        
        return stats
    else:
        print(f"Error: {response.text}")
        return None


def test_rag_stats():
    """Test getting RAG pipeline statistics"""
    print_section("5. RAG Pipeline Statistics")
    
    response = requests.get(f"{BASE_URL}/api/rag/stats")
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print("\nRAG Pipeline Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return stats
    else:
        print(f"Error: {response.text}")
        return None


def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "🧪 " * 20)
    print("  RAG API Test Suite")
    print("🧪 " * 20)
    
    try:
        # Test 1: Health check
        health = test_health_check()
        
        if not health.get('rag_initialized'):
            print("\n⚠️  Warning: RAG system not initialized. Some tests may fail.")
        
        # Test 2: Query the system
        test_query("What is machine learning?", user_id="user1")
        test_query("Explain neural networks", user_id="user1")
        test_query("What is deep learning?", user_id="user2")
        
        # Test 3: Get query history
        test_query_history(limit=10)
        
        # Test 4: Get statistics
        test_statistics()
        
        # Test 5: Get RAG stats
        test_rag_stats()
        
        print_section("✅ All Tests Complete")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Make sure the server is running:")
        print("  uvicorn api_server:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def interactive_mode():
    """Interactive mode for testing queries"""
    print_section("Interactive Query Mode")
    print("\nType your questions (or 'quit' to exit):")
    
    user_id = input("\nEnter your user ID (default: interactive_user): ").strip()
    if not user_id:
        user_id = "interactive_user"
    
    while True:
        print("\n" + "-" * 60)
        question = input("\nQuestion: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        result = test_query(question, user_id)
        
        if result:
            print(f"\n📝 Answer:\n{result['answer']}")


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            interactive_mode()
        elif sys.argv[1] == 'query':
            if len(sys.argv) > 2:
                question = ' '.join(sys.argv[2:])
                test_query(question)
            else:
                print("Usage: python test_api_client.py query <your question>")
        else:
            print("Unknown command. Use 'interactive' or 'query'")
    else:
        run_all_tests()


if __name__ == "__main__":
    main()

