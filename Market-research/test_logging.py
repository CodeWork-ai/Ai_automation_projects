#!/usr/bin/env python3
"""
Test script for the enhanced logging system
"""

import requests
import time
import json

API_BASE_URL = "http://127.0.0.1:8000"

def test_logging_system():
    """Test the enhanced logging system"""
    print("🧪 Testing Enhanced Logging System")
    print("=" * 50)
    
    try:
        # Test 1: Check if server is running
        print("1. Checking server status...")
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
            return
        
        # Test 2: Clear existing logs
        print("\n2. Clearing existing logs...")
        response = requests.delete(f"{API_BASE_URL}/logs")
        if response.status_code == 200:
            print("✅ Logs cleared")
        
        # Test 3: Start a research task to generate logs
        print("\n3. Starting research task to generate logs...")
        research_data = {
            "competitors": {
                "Example Company": {
                    "url": "https://example.com",
                    "content_selectors": ["body"],
                    "exclude_selectors": []
                }
            },
            "model_name": "t5-small"
        }
        
        response = requests.post(f"{API_BASE_URL}/start-research", json=research_data)
        if response.status_code == 200:
            task_data = response.json()
            task_id = task_data["task_id"]
            print(f"✅ Research task started: {task_id}")
            
            # Monitor logs for 30 seconds
            print("\n4. Monitoring logs for 30 seconds...")
            for i in range(15):  # Check every 2 seconds for 30 seconds
                time.sleep(2)
                response = requests.get(f"{API_BASE_URL}/logs")
                if response.status_code == 200:
                    log_data = response.json()
                    if "logs" in log_data:
                        print(f"📊 Logs count: {len(log_data['logs'])}")
                        if log_data["logs"]:
                            latest_log = log_data["logs"][-1]
                            print(f"📝 Latest: [{latest_log['level']}] {latest_log['message'][:60]}...")
                    else:
                        print(f"⚠️ Unexpected log format: {log_data}")
                else:
                    print(f"❌ Failed to fetch logs: {response.status_code}")
        else:
            print(f"❌ Failed to start research: {response.status_code}")
    
    except Exception as e:
        print(f"💥 Test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")
    print("\n💡 To see the logs in action:")
    print("   1. Open your browser to http://127.0.0.1:8000")
    print("   2. Go to the Backend Logs section")
    print("   3. Start a research task")
    print("   4. Watch the detailed logs appear in real-time!")

if __name__ == "__main__":
    test_logging_system()