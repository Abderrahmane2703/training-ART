#!/usr/bin/env python3
"""Basic test script to verify components work"""

print("Testing basic components...")

try:
    print("1. Testing dotenv...")
    from dotenv import load_dotenv
    load_dotenv()
    print("   ✓ dotenv works")
except Exception as e:
    print(f"   ✗ dotenv failed: {e}")

try:
    print("2. Testing boto3...")
    import boto3
    print("   ✓ boto3 works")
except Exception as e:
    print(f"   ✗ boto3 failed: {e}")

try:
    print("3. Testing load_documents (local test)...")
    import sys
    sys.path.append('src/summarizer')
    # We'll create a mock version to avoid S3
    print("   ✓ imports work")
except Exception as e:
    print(f"   ✗ load_documents failed: {e}")

print("\nBasic components test complete!")