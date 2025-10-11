#!/usr/bin/env python3
"""
Quick script to create a test API key
"""

from practice_auth import APIKeyManager

# Initialize the API key manager
manager = APIKeyManager()

# Generate a test API key
print("\n🔑 Creating Test API Key...")
print("=" * 60)

raw_key, api_key = manager.generate_api_key(
    name="Frontend Test Key",
    user_id="test-user",
    rate_limit=100,  # 100 requests per minute
    expires_in_days=30  # Expires in 30 days
)

print(f"\n✅ API Key Created Successfully!")
print(f"\n{'─' * 60}")
print(f"📋 API Key Details:")
print(f"{'─' * 60}")
print(f"Name:           {api_key.name}")
print(f"User ID:        {api_key.user_id}")
print(f"Key ID:         {api_key.key_id}")
print(f"Key Prefix:     {api_key.key_prefix}")
print(f"Rate Limit:     {api_key.rate_limit} requests/minute")
print(f"Created:        {api_key.created_at}")
print(f"Expires:        {api_key.expires_at}")
print(f"\n{'─' * 60}")
print(f"🔐 YOUR API KEY (copy this!):")
print(f"{'─' * 60}")
print(f"\n{raw_key}\n")
print(f"{'─' * 60}")
print(f"\n⚠️  IMPORTANT: Save this key now! It won't be shown again.")
print(f"\n📝 To use in frontend:")
print(f"   1. Open frontend_multimodal.html")
print(f"   2. Paste the key in the 'Your API Key' field")
print(f"   3. Click 'Test Connection'")
print(f"\n{'=' * 60}\n")

