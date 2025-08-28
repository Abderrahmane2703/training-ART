# ART Training Setup Guide

## What This Project Does
This project trains a job offer generation agent using ART (Agent Reinforcement Trainer) - a reinforcement learning framework for LLM agents. The agent learns to generate professional job offers in XML format based on job context (title, language, skills).

## Prerequisites
1. **Python 3.12+** (already have via WSL)
2. **uv package manager** (already installed)
3. **AWS credentials** (already configured in .env.example)
4. **OpenAI-compatible API** for judge model (Azure OpenAI already configured)

## Step 1: Install Dependencies
```bash
cd "/mnt/c/Users/wiame/Documents/CAPITALTECH INTERNSHIP 2025/Cursor/training-ART"
export UV_LINK_MODE=copy
uv sync
```

## Step 2: Set Environment Variables
Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```
# Required for training
AZURE_OPENAI_API_KEY=your_actual_key
AZURE_OPENAI_ENDPOINT=your_actual_endpoint
AZURE_DEPLOYMENT_NAME=your_model_name

# Required for S3 data loading
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=eu-west-3

# Optional
WANDB_API_KEY=your_wandb_key  # For metrics logging
OPENPIPE_API_KEY=your_openpipe_key  # For chat logging
```

## Step 3: Fix Code Issues

### A. Fix Model Path in `src/summarizer/train.py`
**Current Issue**: Line 39 references a non-existent local model path
```python
base_model="/mnt/c/Users/abder/Documents/CapitaleTech/Training/Summary-RL/gemma-3-270m-it",
```

**Fix Options**:
1. **Use Hugging Face model** (recommended):
```python
base_model="Qwen/Qwen2.5-3B-Instruct",
```

2. **Or use smaller model for testing**:
```python
base_model="microsoft/DialoGPT-small",
```

### B. Create S3 Bucket and Upload Test Data
You need to create the S3 bucket and test data referenced in `load_documents.py`:

```bash
# Create S3 bucket (run once)
aws s3 mb s3://job-offer-generation --region eu-west-3
```

Create test dataset file `test_data.json`:
```json
[
  {
    "context": {
      "job_title": "Software Engineer",
      "language": "en",
      "skills": ["Python", "JavaScript", "React"]
    }
  },
  {
    "context": {
      "job_title": "Data Scientist",
      "language": "en", 
      "skills": ["Python", "Machine Learning", "SQL"]
    }
  },
  {
    "context": {
      "job_title": "Développeur Full Stack",
      "language": "fr",
      "skills": ["Node.js", "Vue.js", "PostgreSQL"]
    }
  }
]
```

Upload to S3:
```bash
aws s3 cp test_data.json s3://job-offer-generation/datasets/job_offer_dataset.json
```

### C. Fix Judge Completion Function
Update `src/summarizer/get_judge_completion.py` to use Azure OpenAI:
```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

async def get_judge_completion(prompt: str, max_tokens: int = 150) -> str:
    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting judge completion: {e}")
        return '{"answer": "NO"}'  # Default fallback
```

## Step 4: Test Run (Local Mode)
```bash
# Run training locally (small scale test)
uv run python src/summarizer/train.py
```

## Step 5: Monitor Training
- Training runs in epochs with batch processing
- Validates on small dataset and saves best models
- Check console output for:
  - Data loading from S3
  - Model registration
  - Training progress
  - Validation scores

## Step 6: For AWS SageMaker Deployment

### Option A: Use SkyPilot Backend (Original Design)
Uncomment lines 21-25 in `train.py` and comment out LocalBackend lines.
Requires SkyPilot setup with RunPod/AWS credentials.

### Option B: SageMaker Training Job (Recommended for Production)
Create `sagemaker_train.py`:
```python
import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
session = sagemaker.Session()

estimator = PyTorch(
    entry_point='train.py',
    source_dir='src/summarizer',
    role=role,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'batch-size': 10,
        'epochs': 1,
        'max-steps': 100
    }
)

estimator.fit()
```

## Expected Results
- Agent learns to generate properly formatted XML job offers
- Improves across 5 criteria: language consistency, XML format, context inclusion, skill relevance, completeness  
- Best models saved to S3 for later use
- Training metrics logged to WandB (if configured)

## Troubleshooting

### Common Issues:
1. **S3 Access Denied**: Check AWS credentials and bucket permissions
2. **Model Loading Error**: Verify base model path/name
3. **OpenAI API Error**: Check Azure OpenAI credentials and deployment name
4. **Memory Issues**: Reduce batch size or use smaller model

### Validation Commands:
```bash
# Test AWS access
aws s3 ls s3://job-offer-generation/

# Test Python imports
uv run python -c "import art; print('ART imported successfully')"

# Check model availability
uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

## Next Steps for Production
1. Scale up training data
2. Use GPU instances (SageMaker ml.g4dn.xlarge or larger)
3. Implement model serving endpoint
4. Add evaluation metrics dashboard
5. Set up automated retraining pipeline

---
**Note**: This is a reinforcement learning project where the agent learns from experience, not traditional supervised learning. The reward signal comes from automated evaluation of generated job offers against quality criteria.