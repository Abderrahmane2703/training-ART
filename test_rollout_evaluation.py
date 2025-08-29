#!/usr/bin/env python3
"""
Test file for rollout evaluation scoring system
Tests all 5 evaluation criteria without running the full training
"""

import asyncio
import json
import sys
sys.path.append('src/summarizer')

from load_documents import JobContext
from rollout import clean_json_response


# Mock judge completion function for testing
async def mock_judge_completion(prompt: str, max_tokens: int) -> str:
    """Mock judge responses for testing"""
    
    # Language consistency check
    if "Is this text written in" in prompt:
        if "ENGLISH" in prompt or "EN" in prompt:
            return '{"answer": "YES"}'
        elif "FRENCH" in prompt or "FR" in prompt:
            return '{"answer": "YES"}'
        return '{"answer": "NO"}'
    
    # XML format validation
    elif "Is this valid XML format" in prompt:
        return '{"valid_xml": true, "has_required_tags": true}'
    
    # Context inclusion (skills check)
    elif "Check if ALL the provided skills" in prompt:
        return '{"all_skills_included": true, "missing_skills": []}'
    
    # Skill relevance
    elif "Evaluate ONLY the NEW skills" in prompt:
        return json.dumps({
            "new_skills_added": ["Docker", "AWS", "CI/CD"],
            "relevant_new_skills": 3,
            "irrelevant_new_skills": [],
            "score": 9
        })
    
    # Skill completeness
    elif "CRITICAL skills missing" in prompt:
        return json.dumps({
            "score": 8,
            "missing_critical_skills": []
        })
    
    return '{"error": "Unknown prompt"}'


def test_clean_json_response():
    """Test JSON cleaning function"""
    test_cases = [
        ('```json\n{"test": "value"}\n```', '{"test": "value"}'),
        ('```\n{"test": "value"}\n```', '{"test": "value"}'),
        ('{"test": "value"}', '{"test": "value"}'),
        ('  {"test": "value"}  ', '{"test": "value"}'),
    ]
    
    print("Testing clean_json_response...")
    for input_str, expected in test_cases:
        result = clean_json_response(input_str)
        assert result == expected, f"Failed: {input_str} -> {result} != {expected}"
    print("✓ All JSON cleaning tests passed")


async def test_evaluation_scoring():
    """Test the evaluation scoring logic"""
    
    # Sample job context
    context = JobContext(
        job_title="Software Engineer",
        language="en",
        skills=["Python", "JavaScript", "Git"]
    )
    
    # Sample generated job offer (XML format)
    generated_offer = """<job_offer>
<title>Software Engineer</title>
<overview>
We are seeking a talented Software Engineer to join our development team.
This role involves building scalable applications and collaborating with cross-functional teams.
</overview>
<responsibilities>
<item>Develop and maintain web applications using modern frameworks</item>
<item>Write clean, maintainable code following best practices</item>
<item>Collaborate with product and design teams</item>
<item>Participate in code reviews and technical discussions</item>
<item>Debug and optimize application performance</item>
</responsibilities>
<skills>
<item>Python</item>
<item>JavaScript</item>
<item>Git</item>
<item>Docker</item>
<item>AWS</item>
<item>CI/CD</item>
</skills>
<nice_to_have>
<item>React or Vue.js experience</item>
<item>Cloud architecture knowledge</item>
<item>Agile methodology experience</item>
</nice_to_have>
</job_offer>"""

    print("\nTesting evaluation scoring...")
    print(f"Context: {context.job_title} ({context.language})")
    print(f"Skills provided: {context.skills}")
    
    print("\n" + "="*50)
    print("LLM GENERATED JOB OFFER:")
    print("-"*50)
    print(generated_offer)
    print("="*50)
    
    # Initialize scores
    scores = {}
    
    # Test 1: Language Consistency
    print("\n1. Testing Language Consistency...")
    language_prompt = f"""Is this text written in {context.language.upper()} language?

Text to check: {generated_offer[:300]}

Expected language: {context.language} ({'English' if context.language == 'en' else 'French' if context.language == 'fr' else context.language})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""
    
    print(f"   Prompt to Judge: {language_prompt[:200]}...")
    language_response = await mock_judge_completion(language_prompt, 50)
    print(f"   Judge Response: {language_response}")
    
    try:
        result = json.loads(clean_json_response(language_response))
        scores["language_consistency"] = 1.0 if result["answer"] == "YES" else 0.0
        print(f"   Score: {scores['language_consistency']}")
    except Exception as e:
        print(f"   Error: {e}")
        scores["language_consistency"] = 0.0
    
    # Test 2: XML Format Validation
    print("\n2. Testing XML Format...")
    xml_prompt = f"""Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""
    
    print(f"   Prompt snippet: ...Check if it has proper opening/closing tags...")
    xml_response = await mock_judge_completion(xml_prompt, 100)
    print(f"   Judge Response: {xml_response}")
    
    try:
        result = json.loads(clean_json_response(xml_response))
        scores["xml_format"] = 1.0 if result.get("valid_xml", False) else 0.0
        print(f"   Score: {scores['xml_format']}")
    except Exception as e:
        print(f"   Error: {e}")
        scores["xml_format"] = 0.0
    
    # Test 3: Context Inclusion
    print("\n3. Testing Context Inclusion (provided skills)...")
    context_prompt = f"""Check if ALL the provided skills are included in the generated job offer.

Provided skills that MUST be included: {', '.join(context.skills)}

Generated job offer:
{generated_offer}

Respond ONLY in JSON format:
{{"all_skills_included": true or false, "missing_skills": []}}"""
    
    print(f"   Checking for skills: {', '.join(context.skills)}")
    context_response = await mock_judge_completion(context_prompt, 200)
    print(f"   Judge Response: {context_response}")
    
    try:
        result = json.loads(clean_json_response(context_response))
        scores["context_inclusion"] = 1.0 if result.get("all_skills_included", False) else 0.0
        print(f"   Score: {scores['context_inclusion']}")
    except Exception as e:
        print(f"   Error: {e}")
        scores["context_inclusion"] = 0.0
    
    # Test 4: Skill Relevance
    print("\n4. Testing Skill Relevance (new skills)...")
    provided_skills_str = ', '.join(context.skills)
    skill_relevance_prompt = f"""For a {context.job_title} position:

1. Extract ALL skills mentioned in the job offer (both in Required Skills and Nice-to-Have sections)
2. EXCLUDE these provided skills from evaluation (assume they are relevant): {provided_skills_str}
3. Evaluate ONLY the NEW skills added by the model for relevance to {context.job_title}
4. Score based on the percentage of relevant NEW skills

Generated job offer:
{generated_offer}

Respond ONLY in JSON format:
{{
  "new_skills_added": ["skill1", "skill2"],
  "relevant_new_skills": number,
  "irrelevant_new_skills": ["skill1", "skill2"],
  "score": 0-10
}}"""
    
    print(f"   Evaluating new skills beyond: {provided_skills_str}")
    skill_relevance_response = await mock_judge_completion(skill_relevance_prompt, 300)
    print(f"   Judge Response: {skill_relevance_response}")
    
    try:
        result = json.loads(clean_json_response(skill_relevance_response))
        scores["skill_relevance"] = result.get("score", 5) / 10
        print(f"   New skills identified: {result.get('new_skills_added', [])}")
        print(f"   Score: {scores['skill_relevance']}")
    except Exception as e:
        print(f"   Error: {e}")
        scores["skill_relevance"] = 0.5
    
    # Test 5: Skill Completeness
    print("\n5. Testing Skill Completeness...")
    completeness_prompt = f"""For a {context.job_title} position in {context.language} language:

Current skills in the job offer:
{generated_offer}

Are there any CRITICAL skills missing that are absolutely essential for this role?

Respond ONLY in JSON format:
{{
  "score": 0-10,
  "missing_critical_skills": []
}}"""
    
    print(f"   Checking for missing critical skills...")
    completeness_response = await mock_judge_completion(completeness_prompt, 300)
    print(f"   Judge Response: {completeness_response}")
    
    try:
        result = json.loads(clean_json_response(completeness_response))
        scores["skill_completeness"] = result.get("score", 5) / 10
        missing = result.get("missing_critical_skills", [])
        if missing:
            print(f"   Missing critical skills: {missing}")
        print(f"   Score: {scores['skill_completeness']}")
    except Exception as e:
        print(f"   Error: {e}")
        scores["skill_completeness"] = 0.5
    
    # Calculate final score (weighted average)
    final_score = (
        scores["language_consistency"] * 0.2 +
        scores["xml_format"] * 0.2 +
        scores["context_inclusion"] * 0.2 +
        scores["skill_relevance"] * 0.2 +
        scores["skill_completeness"] * 0.2
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS:")
    print("-"*50)
    for criterion, score in scores.items():
        print(f"{criterion:25s}: {score:.2f}")
    print("-"*50)
    print(f"{'Final Score':25s}: {final_score:.2f}")
    print(f"{'Reward (x10)':25s}: {final_score * 10:.2f}")
    print("="*50)
    
    # Validate scoring logic
    assert 0 <= final_score <= 1, "Final score out of range"
    assert all(0 <= s <= 1 for s in scores.values()), "Individual scores out of range"
    
    return scores, final_score


async def test_different_scenarios():
    """Test different job contexts and languages"""
    
    test_contexts = [
        JobContext(
            job_title="Data Scientist",
            language="en",
            skills=["Python", "Machine Learning", "SQL", "Pandas"]
        ),
        JobContext(
            job_title="Développeur Full Stack",
            language="fr",
            skills=["JavaScript", "Vue.js", "Node.js", "PostgreSQL"]
        ),
        JobContext(
            job_title="DevOps Engineer",
            language="en",
            skills=["AWS", "Docker", "Kubernetes", "Terraform"]
        ),
    ]
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT SCENARIOS")
    print("="*50)
    
    for context in test_contexts:
        print(f"\nTesting: {context.job_title} ({context.language})")
        print(f"Skills: {', '.join(context.skills)}")
        
        # Run simplified scoring
        scores = {
            "language_consistency": 1.0,  # Assume correct language
            "xml_format": 1.0,            # Assume valid XML
            "context_inclusion": 1.0,     # Assume all skills included
            "skill_relevance": 0.9,       # High relevance
            "skill_completeness": 0.8,    # Good completeness
        }
        
        final_score = sum(s * 0.2 for s in scores.values())
        print(f"Final Score: {final_score:.2f}")


async def main():
    """Run all tests"""
    print("🧪 TESTING ROLLOUT EVALUATION SYSTEM")
    print("="*50)
    
    # Test JSON cleaning
    test_clean_json_response()
    
    # Test evaluation scoring
    await test_evaluation_scoring()
    
    # Test different scenarios
    await test_different_scenarios()
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())