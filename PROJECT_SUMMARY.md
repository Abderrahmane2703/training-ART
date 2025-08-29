# Job Offer Generation Training System - Complete Documentation

## Project Overview
Transformed a document summarization system into a job offer generation system using reinforcement learning with the Gemma model.

## Dataset Configuration

### Source
- **Location**: S3 bucket `job-offer-generation`
- **File**: `datasets/job_offer_dataset.json`
- **Size**: 534 samples (53 validation, 481 training)
- **Structure**:
```json
{
  "context": {
    "job_title": "Software Engineer",
    "language": "en",
    "skills": ["Python", "JavaScript", "Git"]
  }
}
```

### Distribution
- Languages: English (en) and French (fr)
- Industries: Tech (30%), Healthcare (15%), Finance (15%), Manufacturing (10%), etc.
- Skills: 3-5 provided per context, model must add relevant missing ones

## File Structure and Changes

### 1. **load_documents.py**
- **Purpose**: Loads job contexts from S3
- **Key Changes**:
  - Replaced `Document` class with `JobContext`
  - Removed fallback sample data
  - Direct S3 loading using boto3
  - Auto-splits: 10% validation, 90% training

### 2. **rollout.py**
- **Purpose**: Generates job offers and evaluates them
- **Key Components**:

#### Job Offer Template
```
{{JOB_TITLE}}
Overview: {{ONE_OR_TWO_SENTENCES_OVERVIEW}}
Key Responsibilities: 5-7 bullet points with action verbs
Required Skills & Qualifications: 5 items including provided + new
Nice-to-Have: 3 items
```

#### Evaluation Criteria (5 Judge Calls)
1. **Language Consistency (20%)**: Checks if output matches input language
2. **XML Format (20%)**: Validates XML structure
3. **Context Inclusion (20%)**: Verifies provided skills included
4. **Skill Relevance (20%)**: Only evaluates NEW skills added by model
5. **Skill Completeness (20%)**: Checks for missing critical skills

#### JSON Response Format
All judge responses use JSON:
```json
{
  "answer": "YES/NO",
  "score": 0-10,
  "missing_skills": [],
  "irrelevant_skills": []
}
```

### 3. **train.py**
- **Purpose**: Training loop with validation-based checkpoint saving
- **Key Features**:
  - Base model: `./gemma-3-270m-it` (local) or HuggingFace model
  - Batch size: 10 contexts
  - Validation: 2 attempts per context
  - Training: 10 attempts per context
  - **Smart Saving**: Only saves to S3 when validation score improves
  - No early stopping (removed per request)

#### Training Process
```python
# Per batch:
1. Generate 106 validation trajectories (53 contexts × 2)
2. Generate 100 training trajectories (10 contexts × 10)
3. Calculate validation score
4. Train model on training trajectories
5. Save to S3 ONLY if validation improved
```

### 4. **get_judge_completion.py**
- **Purpose**: LLM judge for evaluation
- **Configuration**: Azure OpenAI
```python
# Required environment variables:
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_DEPLOYMENT_NAME=gpt-35-turbo
```

## Training Mechanics

### Reinforcement Learning Process
1. **Each trajectory gets individual reward** (0-10)
2. **High rewards** → Increase probability of those patterns
3. **Low rewards** → Decrease probability of those patterns
4. **Weight updates**: After each batch (48 updates per epoch)

### Example Reward Calculation
- Context: `{job_title: "Data Scientist", language: "en", skills: ["Python", "SQL"]}`
- Model generates job offer
- Scores: Language ✓ (1.0), XML ✓ (1.0), Skills included ✓ (1.0), Relevance (0.9), Completeness (0.7)
- Final reward: (1+1+1+0.9+0.7)/5 × 10 = **9.2/10**

### Training Numbers
- **Per batch**: 100 training + 106 validation = 206 job offers
- **Per epoch**: 48 batches × 206 = ~9,888 job offers
- **Judge calls**: 5 per job offer = ~49,440 API calls per epoch

## S3 Storage Structure
```
s3://job-offer-generation/
├── datasets/
│   └── job_offer_dataset.json
└── job-offer-generation/
    └── job-offer-agent/
        └── checkpoints/
            └── [model weights - only best saved]
```

## Key Commands

### Test Data Loading
```bash
python quick_test.py
```

### Test Judge
```bash
python test_judge.py
```

### Count Dataset Entries
```bash
python count_json.py dataset.json
```

### Start Training
```bash
python src/summarizer/train.py
```

## Important Notes

1. **Model weights are stored primarily in S3**, not locally
2. **Validation-based saving**: Only best model kept in S3
3. **No averaging of rewards**: Each of 100 trajectories contributes individually
4. **Judge makes 5 calls per job offer** for comprehensive evaluation
5. **Skills evaluation**: Only NEW skills are judged for relevance, not provided ones

## Environment Variables Required
```env
# AWS (for S3)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1

# Azure OpenAI (for judge)
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com
AZURE_DEPLOYMENT_NAME=gpt-35-turbo

# Optional
WANDB_API_KEY=xxx
OPENPIPE_API_KEY=xxx
```

## Next Steps When Resuming
1. Ensure all environment variables are set
2. Run `python test_judge.py` to verify Azure connection
3. Run `python quick_test.py` to verify S3 dataset access
4. Start training with `python src/summarizer/train.py`
5. Monitor validation scores for improvement
6. Best model automatically saved to S3

## Current Status
- ✅ Data loading from S3 working (534 samples)
- ✅ Rollout configured for job offer generation
- ✅ Judge using JSON responses
- ✅ Training loop with validation-based saving
- ✅ Azure OpenAI integration for judge
- ⏳ Ready to start training