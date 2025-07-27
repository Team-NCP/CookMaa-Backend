#!/usr/bin/env python3
"""
Test Railway setup locally
"""

import subprocess
import sys
import os

def test_docker_build():
    """Test if our Dockerfile builds successfully"""
    print("🔍 Testing Docker build...")
    try:
        result = subprocess.run([
            "docker", "build", "-t", "cookma-test", "."
        ], capture_output=True, text=True, cwd="/Users/sudhanvaacharya/Desktop/Code Projects/CookMaa/backend")
        
        if result.returncode == 0:
            print("✅ Docker build successful")
            return True
        else:
            print("❌ Docker build failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def test_docker_run():
    """Test if container starts and responds"""
    print("\n🔍 Testing Docker container startup...")
    try:
        # Start container in background
        process = subprocess.Popen([
            "docker", "run", "-p", "8002:8000", "-e", "PORT=8000", "cookma-test"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a bit for startup
        import time
        time.sleep(5)
        
        # Test if container is responding
        import requests
        try:
            response = requests.get("http://localhost:8002/health", timeout=5)
            if response.status_code == 200:
                print("✅ Container responds to health check")
                print(f"   Response: {response.json()}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to reach container: {e}")
        
        # Get container logs
        process.terminate()
        stdout, stderr = process.communicate()
        print("\n📋 Container logs:")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        
    except Exception as e:
        print(f"❌ Container test error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Railway setup locally...")
    print("=" * 50)
    
    if test_docker_build():
        test_docker_run()
    
    print("\n" + "=" * 50)
    print("🎯 Next steps:")
    print("1. Check Railway build logs in dashboard")
    print("2. Check Railway deploy logs for Python errors")
    print("3. Verify PORT environment variable in Railway")