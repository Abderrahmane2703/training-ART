#!/usr/bin/env python3
"""Test S3 data loading without heavy ART imports"""

print("🔄 Testing S3 data loading...")

import sys
import os
sys.path.append('src/summarizer')

from dotenv import load_dotenv
load_dotenv()

try:
    print("🔄 Testing load_documents function...")
    from load_documents import load_documents
    
    print("🔄 Attempting to load data from S3...")
    val_contexts, train_contexts = load_documents()
    
    print(f"✅ Success! Loaded:")
    print(f"   - {len(train_contexts)} training contexts")
    print(f"   - {len(val_contexts)} validation contexts")
    
    # Show first example
    if train_contexts:
        first = train_contexts[0]
        print(f"   - First example: {first.job_title} ({first.language})")
        
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    print("\nThis likely means:")
    print("1. Missing AWS credentials in .env")
    print("2. Missing S3 dataset at: s3://job-offer-generation/datasets/job_offer_dataset.json")
    print("3. S3 permissions issue")
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("\n✅ .env file exists")
    else:
        print("\n❌ .env file missing - copy from .env.example")