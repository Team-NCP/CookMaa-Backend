#!/usr/bin/env python3

import requests
import json
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.getenv("VAPI_PRIVATE_KEY") or "39a39332-e4d3-412c-9f7f-679ef3963c9f"

def monitor_function_calls():
    """Monitor VAPI calls in real-time to verify function calls are working"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Real-time VAPI Function Call Monitor")
    print("=" * 50)
    print("Watching for function calls...")
    print("Say 'Hey Kukma, next step' to test!")
    print("Press Ctrl+C to stop monitoring")
    print("")
    
    last_call_count = 0
    
    try:
        while True:
            try:
                # Get recent calls
                response = requests.get(
                    "https://api.vapi.ai/call",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    calls = response.json()
                    current_count = len(calls)
                    
                    # Check for new calls
                    if current_count > last_call_count:
                        print(f"📞 Found {current_count - last_call_count} new call(s)")
                        
                        # Check the most recent calls for function calls
                        recent_calls = calls[:5]  # Check last 5 calls
                        
                        for call in recent_calls:
                            call_id = call.get('id', 'unknown')
                            status = call.get('status', 'unknown')
                            
                            # Get detailed call info
                            try:
                                detail_response = requests.get(
                                    f"https://api.vapi.ai/call/{call_id}",
                                    headers=headers,
                                    timeout=10
                                )
                                
                                if detail_response.status_code == 200:
                                    call_details = detail_response.json()
                                    messages = call_details.get('messages', [])
                                    
                                    function_calls_found = []
                                    transcripts_found = []
                                    
                                    for msg in messages:
                                        msg_type = msg.get('type', '')
                                        
                                        if msg_type == 'function-call':
                                            function_name = msg.get('functionCall', {}).get('name', 'unknown')
                                            function_calls_found.append(function_name)
                                            
                                        elif msg_type == 'transcript':
                                            role = msg.get('role', '')
                                            transcript = msg.get('transcript', '')
                                            if role == 'user':
                                                transcripts_found.append(transcript)
                                    
                                    if function_calls_found:
                                        print(f"🎉 FUNCTION CALLS DETECTED in call {call_id[:8]}:")
                                        for func in function_calls_found:
                                            print(f"   ✅ {func}")
                                        print(f"   User said: {transcripts_found[-1] if transcripts_found else 'N/A'}")
                                        print("")
                                    
                                    elif transcripts_found:
                                        print(f"📝 Transcript only (no function call) in call {call_id[:8]}:")
                                        print(f"   User: {transcripts_found[-1]}")
                                        print(f"   ⚠️  No function call triggered")
                                        print("")
                                        
                            except Exception as e:
                                print(f"⚠️  Could not get details for call {call_id}: {e}")
                    
                    last_call_count = current_count
                    
                else:
                    print(f"❌ Failed to get calls: {response.status_code}")
                
            except requests.RequestException as e:
                print(f"⚠️  Network error: {e}")
            
            # Wait before next check
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor_function_calls()