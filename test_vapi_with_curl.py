#!/usr/bin/env python3

import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.getenv("VAPI_PRIVATE_KEY") or "39a39332-e4d3-412c-9f7f-679ef3963c9f"
ASSISTANT_ID = "b9c6dfa6-d816-4af9-b5e8-ac924baf6509"

def test_vapi_responses():
    """Test VAPI assistant responses using direct API calls"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🧪 Testing VAPI Assistant with curl-like requests")
    print("=" * 60)
    
    # Test different messages to see if functions are called
    test_messages = [
        "next step",
        "hey kukma next step", 
        "repeat that",
        "go back",
        "what's for dinner?",  # Non-function call
    ]
    
    for message in test_messages:
        print(f"\n📤 Testing message: '{message}'")
        print("-" * 40)
        
        # Create a test call with the message
        call_payload = {
            "assistantId": ASSISTANT_ID,
            "customer": {
                "number": "+1234567890"
            },
            "assistantOverrides": {
                "firstMessage": f"User said: {message}"
            }
        }
        
        try:
            # Start a call
            print("🚀 Starting VAPI call...")
            response = requests.post(
                "https://api.vapi.ai/call",
                headers=headers,
                json=call_payload,
                timeout=30
            )
            
            if response.status_code == 201:
                call_data = response.json()
                call_id = call_data.get("id", "")
                print(f"✅ Call created: {call_id}")
                
                # Wait a moment for processing
                time.sleep(3)
                
                # Get call details to see what happened
                detail_response = requests.get(
                    f"https://api.vapi.ai/call/{call_id}",
                    headers=headers,
                    timeout=10
                )
                
                if detail_response.status_code == 200:
                    call_details = detail_response.json()
                    messages = call_details.get("messages", [])
                    
                    print(f"📊 Call details retrieved, {len(messages)} messages")
                    
                    function_calls = []
                    transcripts = []
                    assistant_responses = []
                    
                    for msg in messages:
                        msg_type = msg.get("type", "")
                        
                        if msg_type == "function-call":
                            func_name = msg.get("functionCall", {}).get("name", "unknown")
                            function_calls.append(func_name)
                            print(f"🎯 FUNCTION CALL: {func_name}")
                            
                        elif msg_type == "transcript":
                            role = msg.get("role", "")
                            transcript = msg.get("transcript", "")
                            if role == "assistant":
                                assistant_responses.append(transcript)
                                print(f"🤖 Assistant: {transcript}")
                            elif role == "user":
                                transcripts.append(transcript)
                                print(f"👤 User: {transcript}")
                    
                    # Summary
                    if function_calls:
                        print(f"✅ SUCCESS: Function calls detected: {function_calls}")
                    else:
                        print("❌ FAILED: No function calls detected")
                        if assistant_responses:
                            print(f"💭 Instead got response: {assistant_responses[-1]}")
                
                # End the call
                requests.patch(
                    f"https://api.vapi.ai/call/{call_id}",
                    headers=headers,
                    json={"status": "ended"}
                )
                
            else:
                print(f"❌ Failed to create call: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Test error: {e}")
        
        print()  # Add spacing between tests

def test_webhook_directly():
    """Test our webhook directly to verify it works"""
    
    print("\n🔧 Testing webhook directly...")
    print("-" * 40)
    
    # Test function call webhook
    function_call_payload = {
        "message": {
            "type": "function-call",
            "functionCall": {
                "name": "next_step"
            }
        },
        "call": {
            "id": "test-curl-function-call"
        }
    }
    
    try:
        response = requests.post(
            "https://cookmaa-backend-production.up.railway.app/webhook/vapi-message",
            json=function_call_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📡 Webhook function call test: {response.status_code}")
        if response.status_code == 200:
            result = response.json() if response.content else response.text
            print(f"✅ Webhook response: {result}")
        else:
            print(f"❌ Webhook error: {response.text}")
            
    except Exception as e:
        print(f"❌ Webhook test error: {e}")

if __name__ == "__main__":
    test_vapi_responses()
    test_webhook_directly()