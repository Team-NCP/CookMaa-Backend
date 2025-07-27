#!/usr/bin/env python3

"""
Test deployed Railway service
"""

import requests
import sys

def test_deployment(base_url):
    """Test the deployed service"""
    print(f"🔍 Testing deployment at: {base_url}")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working")
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Status: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False
    
    # Test 2: Health check
    print("\n2️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   APIs: {data.get('apis', {})}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test 3: API docs
    print("\n3️⃣ Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API docs accessible")
            print(f"   URL: {base_url}/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Deployment test completed!")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = input("Enter your Railway deployment URL (e.g., https://your-app.railway.app): ").strip()
    
    if not base_url.startswith("http"):
        base_url = "https://" + base_url
    
    test_deployment(base_url)