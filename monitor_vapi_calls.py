#!/usr/bin/env python3

import requests
import time
import json
from datetime import datetime

WEBHOOK_URL = "https://cookmaa-backend-production.up.railway.app"

def monitor_vapi_calls():
    """Monitor VAPI webhook calls for function-call messages"""
    
    print("🎤 Monitoring VAPI webhook calls...")
    print("=" * 60)
    print("Looking for function-call messages (not just transcripts)")
    print("Test by saying 'next step' in the iOS app")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    last_call_count = 0
    
    try:
        while True:
            # Get recent calls
            response = requests.get(f"{WEBHOOK_URL}/debug/vapi-calls", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_count = data.get("recent_calls_count", 0)
                recent_calls = data.get("recent_calls", [])
                
                # Check if we have new calls
                if current_count > last_call_count:
                    print(f"\n🆕 New webhook call detected at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Check the most recent call for function-call messages
                    if recent_calls:
                        latest_call = recent_calls[-1]
                        try:
                            # Parse the JSON in the log entry
                            call_data = json.loads(latest_call.split(": ", 1)[1])
                            message_type = call_data.get("message", {}).get("type", "")
                            
                            if message_type == "function-call":
                                function_name = call_data.get("message", {}).get("functionCall", {}).get("name", "")
                                print(f"✅ FUNCTION CALL DETECTED: {function_name}")
                                print(f"   This means voice announcements should now work!")
                                
                            elif message_type == "transcript":
                                transcript = call_data.get("message", {}).get("transcript", "")
                                role = call_data.get("message", {}).get("role", "")
                                print(f"📝 Transcript ({role}): {transcript}")
                                
                            else:
                                print(f"ℹ️  Other message type: {message_type}")
                                
                        except Exception as e:
                            print(f"⚠️  Could not parse call data: {e}")
                    
                    last_call_count = current_count
                
            else:
                print(f"❌ Failed to get calls: {response.status_code}")
            
            # Wait before next check
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")

if __name__ == "__main__":
    monitor_vapi_calls()