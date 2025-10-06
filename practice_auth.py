"""
PRACTICE TASK #2: Implement API Key Authentication System

OBJECTIVE:
Build a secure API key authentication system to protect your RAG endpoints.
This is essential for production APIs to control access and track usage per user.

LEARNING GOALS:
- Generate secure API keys
- Store hashed credentials safely
- Implement authentication middleware
- Track API usage per key
- Implement rate limiting per user
- Handle authentication errors

DIFFICULTY: Intermediate-Advanced

ESTIMATED TIME: 3-4 hours

INSTRUCTIONS:
Fill in the methods marked with TODO comments. Read the detailed comments to understand
what each method should do, then implement the logic yourself.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import secrets
import sqlite3
from pathlib import Path
import time


@dataclass
class APIKey:
    """
    Represents an API key with its metadata
    
    Attributes:
        key_id: Unique identifier for this key
        name: Human-readable name (e.g., "Production Key", "Development Key")
        key_prefix: First 8 characters of the key (for display)
        key_hash: Hashed version of the full key (for security)
        user_id: ID of the user who owns this key
        created_at: When the key was created
        expires_at: When the key expires (optional)
        is_active: Whether the key is currently active
        rate_limit: Maximum requests per minute
        usage_count: Total number of times used
    """
    key_id: int
    name: str
    key_prefix: str
    key_hash: str
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    rate_limit: int
    usage_count: int


@dataclass
class APIKeyValidation:
    """
    Result of API key validation
    
    Attributes:
        is_valid: Whether the key is valid
        api_key: The API key object if valid
        error_message: Error message if invalid
    """
    is_valid: bool
    api_key: Optional[APIKey] = None
    error_message: Optional[str] = None


class APIKeyManager:
    """
    Manages API key generation, validation, and usage tracking
    
    SECURITY BEST PRACTICES:
    1. Never store raw API keys - only store hashed versions
    2. Use secrets module for cryptographically secure random keys
    3. Implement rate limiting to prevent abuse
    4. Log authentication attempts
    5. Support key expiration and rotation
    
    WHAT YOU NEED TO IMPLEMENT:
    1. Database schema for API keys
    2. Secure key generation
    3. Key hashing and verification
    4. Rate limiting logic
    5. Usage tracking
    6. Key lifecycle management (create, revoke, rotate)
    """
    
    def __init__(self, db_path: str = "./data/api_keys.db"):
        """
        Initialize the API Key Manager
        
        Args:
            db_path: Path to SQLite database
        
        TODO: Implement initialization
        STEPS:
        1. Store db_path as instance variable
        2. Call self._init_database() to set up tables
        """
        # TODO: Implement initialization
        pass
    
    def _init_database(self) -> None:
        """
        Create database tables for API keys and usage tracking
        
        TABLES TO CREATE:
        
        1. api_keys table:
           - key_id: INTEGER PRIMARY KEY AUTOINCREMENT
           - name: TEXT NOT NULL (human-readable name)
           - key_prefix: TEXT NOT NULL (first 8 chars for display)
           - key_hash: TEXT NOT NULL UNIQUE (SHA-256 hash of full key)
           - user_id: TEXT NOT NULL (owner of the key)
           - created_at: DATETIME DEFAULT CURRENT_TIMESTAMP
           - expires_at: DATETIME (nullable)
           - is_active: BOOLEAN DEFAULT 1
           - rate_limit: INTEGER DEFAULT 60 (requests per minute)
           - usage_count: INTEGER DEFAULT 0
        
        2. api_key_usage table (for rate limiting):
           - id: INTEGER PRIMARY KEY AUTOINCREMENT
           - key_id: INTEGER (foreign key)
           - timestamp: DATETIME DEFAULT CURRENT_TIMESTAMP
           - endpoint: TEXT (which endpoint was called)
           - success: BOOLEAN (was request successful)
        
        TODO: Implement database initialization
        STEPS:
        1. Create parent directory if it doesn't exist
        2. Connect to SQLite database
        3. Create api_keys table with schema above
        4. Create api_key_usage table with schema above
        5. Create indexes for faster queries:
           - Index on key_hash for quick lookups
           - Index on timestamp for rate limiting queries
        6. Commit and close connection
        """
        # TODO: Implement database setup
        # HINT: Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # HINT: conn = sqlite3.connect(self.db_path)
        # HINT: cursor.execute("CREATE TABLE IF NOT EXISTS ...")
        pass
    
    def generate_api_key(
        self,
        name: str,
        user_id: str,
        rate_limit: int = 60,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key
        
        WHY THIS MATTERS:
        API keys must be cryptographically secure random strings. They should
        be unpredictable and have enough entropy to prevent guessing attacks.
        
        KEY FORMAT: "rag_" + 32 random hex characters (e.g., "rag_a1b2c3d4...")
        
        Args:
            name: Human-readable name for the key
            user_id: User who owns this key
            rate_limit: Maximum requests per minute
            expires_in_days: Days until expiration (None = no expiration)
        
        Returns:
            Tuple of (raw_api_key, APIKey object)
            Note: Raw key is only returned once - must be saved by user
        
        TODO: Implement key generation
        STEPS:
        1. Generate random key:
           - Use secrets.token_hex(16) to get 32 hex characters
           - Prefix with "rag_" to identify your keys
           - Full key: f"rag_{secrets.token_hex(16)}"
        
        2. Extract key prefix (first 8 chars) for display
        
        3. Hash the full key:
           - Use SHA-256: hashlib.sha256(full_key.encode()).hexdigest()
           - Never store the raw key in database!
        
        4. Calculate expiration:
           - If expires_in_days is provided:
             expires_at = datetime.now() + timedelta(days=expires_in_days)
           - Else: expires_at = None
        
        5. Insert into database:
           - INSERT INTO api_keys (name, key_prefix, key_hash, user_id,
             expires_at, rate_limit) VALUES (...)
           - Get the last inserted row ID
        
        6. Create APIKey object with all fields
        
        7. Return (raw_key, api_key_object)
        """
        # TODO: Implement key generation
        # HINT: secrets.token_hex(16) generates secure random hex string
        # HINT: hashlib.sha256(string.encode()).hexdigest() for hashing
        pass
    
    def validate_api_key(self, api_key: str) -> APIKeyValidation:
        """
        Validate an API key
        
        VALIDATION CHECKS:
        1. Key format is correct (starts with "rag_")
        2. Key exists in database (hashed version matches)
        3. Key is active (is_active = True)
        4. Key has not expired (expires_at > now or is None)
        5. Rate limit not exceeded
        
        Args:
            api_key: The raw API key to validate
        
        Returns:
            APIKeyValidation object with validation result
        
        TODO: Implement key validation
        STEPS:
        1. Check key format:
           - Must start with "rag_"
           - If not, return APIKeyValidation(is_valid=False, error_message="Invalid key format")
        
        2. Hash the provided key:
           - key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        3. Query database:
           - SELECT * FROM api_keys WHERE key_hash = ? AND is_active = 1
           - If no result, return APIKeyValidation(is_valid=False, error_message="Invalid API key")
        
        4. Check expiration:
           - If expires_at is not None and expires_at < datetime.now():
             return APIKeyValidation(is_valid=False, error_message="API key expired")
        
        5. Check rate limit (call self._check_rate_limit(key_id)):
           - If exceeded, return APIKeyValidation(is_valid=False, error_message="Rate limit exceeded")
        
        6. If all checks pass:
           - Create APIKey object from database row
           - Return APIKeyValidation(is_valid=True, api_key=api_key_obj)
        """
        # TODO: Implement validation
        pass
    
    def _check_rate_limit(self, key_id: int, rate_limit: int) -> bool:
        """
        Check if API key has exceeded rate limit
        
        RATE LIMITING STRATEGY:
        Count requests in the last 60 seconds. If count >= rate_limit, reject.
        
        Args:
            key_id: ID of the API key
            rate_limit: Maximum requests per minute
        
        Returns:
            True if within limit, False if exceeded
        
        TODO: Implement rate limit checking
        STEPS:
        1. Calculate time window:
           - one_minute_ago = datetime.now() - timedelta(minutes=1)
        
        2. Count recent requests:
           - SELECT COUNT(*) FROM api_key_usage
             WHERE key_id = ? AND timestamp > ?
           - Get the count
        
        3. Compare count with rate_limit:
           - If count >= rate_limit: return False (exceeded)
           - Else: return True (within limit)
        """
        # TODO: Implement rate limit check
        # HINT: Use datetime.now() - timedelta(minutes=1) for time window
        pass
    
    def record_usage(
        self,
        api_key: APIKey,
        endpoint: str,
        success: bool
    ) -> None:
        """
        Record API key usage for tracking and rate limiting
        
        WHY THIS IS IMPORTANT:
        - Enables rate limiting
        - Provides usage analytics
        - Helps detect abuse patterns
        - Supports billing/quota systems
        
        Args:
            api_key: The API key object
            endpoint: Which endpoint was called (e.g., "/api/query")
            success: Whether the request was successful
        
        TODO: Implement usage recording
        STEPS:
        1. Insert usage record:
           - INSERT INTO api_key_usage (key_id, endpoint, success)
             VALUES (?, ?, ?)
        
        2. Update usage count:
           - UPDATE api_keys SET usage_count = usage_count + 1
             WHERE key_id = ?
        
        3. Commit changes
        """
        # TODO: Implement usage recording
        pass
    
    def revoke_api_key(self, key_id: int) -> bool:
        """
        Revoke (deactivate) an API key
        
        USE CASES:
        - Key compromised
        - User no longer needs access
        - Security incident
        
        Args:
            key_id: ID of the key to revoke
        
        Returns:
            True if successful, False if key not found
        
        TODO: Implement key revocation
        STEPS:
        1. Update database:
           - UPDATE api_keys SET is_active = 0 WHERE key_id = ?
        
        2. Check if any rows were affected:
           - If cursor.rowcount > 0: return True
           - Else: return False (key not found)
        """
        # TODO: Implement revocation
        pass
    
    def get_user_keys(self, user_id: str) -> List[APIKey]:
        """
        Get all API keys for a user
        
        SECURITY NOTE:
        Never return the full key or key_hash. Only return key_prefix
        so users can identify their keys.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of APIKey objects (without full keys)
        
        TODO: Implement key listing
        STEPS:
        1. Query database:
           - SELECT * FROM api_keys WHERE user_id = ?
             ORDER BY created_at DESC
        
        2. Convert each row to APIKey object
        
        3. Return list of APIKey objects
        """
        # TODO: Implement key listing
        pass
    
    def get_usage_stats(self, key_id: int) -> Dict[str, Any]:
        """
        Get usage statistics for an API key
        
        METRICS TO CALCULATE:
        - Total requests
        - Successful requests
        - Failed requests
        - Success rate (percentage)
        - Requests per endpoint
        - Recent activity (last 24 hours)
        
        Args:
            key_id: ID of the API key
        
        Returns:
            Dictionary with usage statistics
        
        TODO: Implement statistics calculation
        STEPS:
        1. Get total requests:
           - SELECT COUNT(*) FROM api_key_usage WHERE key_id = ?
        
        2. Get successful requests:
           - SELECT COUNT(*) FROM api_key_usage WHERE key_id = ? AND success = 1
        
        3. Calculate success rate:
           - success_rate = (successful / total * 100) if total > 0 else 0
        
        4. Get requests by endpoint:
           - SELECT endpoint, COUNT(*) as count FROM api_key_usage
             WHERE key_id = ? GROUP BY endpoint
        
        5. Get recent activity (last 24 hours):
           - SELECT COUNT(*) FROM api_key_usage
             WHERE key_id = ? AND timestamp > datetime('now', '-1 day')
        
        6. Return dictionary with all stats
        """
        # TODO: Implement statistics
        pass


# ============================================================================
# FastAPI Integration Example (Already implemented)
# ============================================================================

"""
HOW TO USE THIS WITH FASTAPI:

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader

app = FastAPI()
api_key_manager = APIKeyManager()

# Define API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)):
    '''Dependency to validate API keys'''
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    validation = api_key_manager.validate_api_key(api_key)
    
    if not validation.is_valid:
        raise HTTPException(status_code=401, detail=validation.error_message)
    
    # Record usage
    api_key_manager.record_usage(
        validation.api_key,
        endpoint="/api/query",
        success=True
    )
    
    return validation.api_key

@app.post("/api/query")
async def query_rag(
    request: QueryRequest,
    api_key: APIKey = Depends(validate_api_key)
):
    '''Protected endpoint - requires valid API key'''
    # Process query...
    return {"answer": "..."}

@app.post("/api/keys")
async def create_api_key(name: str, user_id: str):
    '''Create a new API key'''
    raw_key, api_key_obj = api_key_manager.generate_api_key(
        name=name,
        user_id=user_id,
        rate_limit=60,
        expires_in_days=365
    )
    
    return {
        "api_key": raw_key,  # Only shown once!
        "key_id": api_key_obj.key_id,
        "expires_at": api_key_obj.expires_at
    }
"""


# ============================================================================
# Testing Code (Run this after implementing the above)
# ============================================================================

def test_api_key_manager():
    """
    Test your API key manager implementation
    
    RUN THIS AFTER IMPLEMENTING THE ABOVE METHODS
    """
    print("Testing API Key Manager Implementation...")
    
    # Create manager
    manager = APIKeyManager(db_path="./data/test_api_keys.db")
    
    # Test 1: Generate API key
    print("\n1. Testing key generation...")
    raw_key, api_key = manager.generate_api_key(
        name="Test Key",
        user_id="user123",
        rate_limit=10
    )
    assert raw_key.startswith("rag_"), "Failed: Key should start with 'rag_'"
    assert len(raw_key) == 36, "Failed: Key should be 36 characters"
    print(f"✓ Generated key: {raw_key[:12]}...")
    
    # Test 2: Validate valid key
    print("\n2. Testing key validation...")
    validation = manager.validate_api_key(raw_key)
    assert validation.is_valid, "Failed: Valid key should pass validation"
    print("✓ Key validation works")
    
    # Test 3: Reject invalid key
    print("\n3. Testing invalid key rejection...")
    validation = manager.validate_api_key("invalid_key")
    assert not validation.is_valid, "Failed: Invalid key should be rejected"
    print("✓ Invalid key rejection works")
    
    # Test 4: Rate limiting
    print("\n4. Testing rate limiting...")
    for i in range(12):  # Exceed rate limit of 10
        manager.record_usage(api_key, "/test", True)
    
    validation = manager.validate_api_key(raw_key)
    assert not validation.is_valid, "Failed: Should be rate limited"
    print("✓ Rate limiting works")
    
    # Test 5: Key revocation
    print("\n5. Testing key revocation...")
    success = manager.revoke_api_key(api_key.key_id)
    assert success, "Failed: Should successfully revoke key"
    
    validation = manager.validate_api_key(raw_key)
    assert not validation.is_valid, "Failed: Revoked key should be invalid"
    print("✓ Key revocation works")
    
    print("\n✅ All tests passed! Your authentication system is working!")
    
    # Cleanup
    import os
    if os.path.exists("./data/test_api_keys.db"):
        os.remove("./data/test_api_keys.db")


# UNCOMMENT THIS TO TEST YOUR IMPLEMENTATION:
# if __name__ == "__main__":
#     test_api_key_manager()

