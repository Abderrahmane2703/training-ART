import art
import openai
import random
import json
from pydantic import BaseModel
import time
import os

from get_judge_completion import get_judge_completion
from load_documents import JobContext

from openpipe.client import OpenPipe


op_client = OpenPipe()


class JobOfferScenario(BaseModel):
    context: JobContext
    step: int = 0


def clean_json_response(response: str) -> str:
    """Clean JSON response from markdown code blocks."""
    clean = response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(model: art.Model, scenario: JobOfferScenario) -> art.Trajectory:
    client = model.openai_client()

    # Job offer template
    template = """{{JOB_TITLE}}
Overview
{{ONE_OR_TWO_SENTENCES_OVERVIEW}}

Key Responsibilities
* {{RESPONSIBILITY_1}}
* {{RESPONSIBILITY_2}}
* {{RESPONSIBILITY_3}}
* {{RESPONSIBILITY_4}}
* {{RESPONSIBILITY_5}}

Required Skills & Qualifications
* {{REQUIRED_SKILL_1}}
* {{REQUIRED_SKILL_2}}
* {{REQUIRED_SKILL_3}}
* {{REQUIRED_SKILL_4}}
* {{REQUIRED_SKILL_5}}

Nice-to-Have
* {{NICE_TO_HAVE_1}}
* {{NICE_TO_HAVE_2}}
* {{NICE_TO_HAVE_3}}

Guidelines:
- Overview: 2-3 sentences describing the purpose of the role and its impact on the company
- Key Responsibilities: List 5-7 bullet points with action verbs (e.g., "Develop", "Manage", "Lead", "Optimize")
- Focus on outcomes and accountability, not just tasks
- Skills: Include provided skills and add relevant missing ones"""

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"""You are a specialized AI assistant that generates professional job offers in XML format.
You must follow this template structure and output valid XML.

Template:
{template}
""",
            }
        ],
        reward=0,
        metrics={
            "language_consistency": 0,
            "xml_format": 0,
            "context_inclusion": 0,
            "skill_relevance": 0,
            "skill_completeness": 0,
            "total_score": 0,
        },
    )

    # Build the generation prompt
    context_info = f"""Job Title: {scenario.context.job_title}
Language: {scenario.context.language}"""
    
    if scenario.context.skills:
        context_info += f"\nProvided Skills: {', '.join(scenario.context.skills)}"
    
    generation_prompt = f"""Generate a complete job offer based on this context:

{context_info}

Instructions:
1. Use the same language as provided ({scenario.context.language})
2. Include all provided skills and add relevant ones that are missing
3. Create 5-7 key responsibilities using action verbs
4. Output in valid XML format with these tags: <job_offer>, <title>, <overview>, <responsibilities>, <skills>, <nice_to_have>
5. Each responsibility, skill, and nice-to-have should be in its own tag

Generate the job offer now:"""

    trajectory.messages_and_choices.append(
        {"role": "user", "content": generation_prompt}
    )

    requested_at = int(time.time() * 1000)

    # Generate job offer
    messages = trajectory.messages()
    completion = await client.chat.completions.create(
        model=model.inference_model_name, messages=messages, max_tokens=1500
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)
    generated_offer = choice.message.content

    # Initialize scoring
    total_criteria = 5
    scores = {}

    # Criterion 1: Language Consistency
    language_prompt = f"""Is this text written in {scenario.context.language.upper()} language?

Text to check: {generated_offer[:300]}

Expected language: {scenario.context.language} ({'English' if scenario.context.language == 'en' else 'French' if scenario.context.language == 'fr' else scenario.context.language})

Respond ONLY in JSON format:
{{"answer": "YES" or "NO"}}"""

    language_response = await get_judge_completion(language_prompt, max_tokens=50)
    try:
        result = json.loads(clean_json_response(language_response))
        scores["language_consistency"] = 1.0 if result["answer"] == "YES" else 0.0
    except:
        scores["language_consistency"] = 0.0  # Default to 0 if parsing fails

    # Criterion 2: XML Format Validation
    xml_prompt = f"""Is this valid XML format? Check if it has proper opening/closing tags.

Text to check:
{generated_offer}

Respond ONLY in JSON format:
{{"valid_xml": true or false, "has_required_tags": true or false}}"""

    xml_response = await get_judge_completion(xml_prompt, max_tokens=100)
    try:
        result = json.loads(clean_json_response(xml_response))
        scores["xml_format"] = 1.0 if result.get("valid_xml", False) else 0.0
    except:
        scores["xml_format"] = 0.0

    # Criterion 3: Context Inclusion (check if provided skills are included)
    if scenario.context.skills:
        context_prompt = f"""Check if ALL the provided skills are included in the generated job offer.

Provided skills that MUST be included: {', '.join(scenario.context.skills)}

Generated job offer:
{generated_offer}

Respond ONLY in JSON format:
{{"all_skills_included": true or false, "missing_skills": []}}"""

        context_response = await get_judge_completion(context_prompt, max_tokens=200)
        try:
            result = json.loads(clean_json_response(context_response))
            scores["context_inclusion"] = 1.0 if result.get("all_skills_included", False) else 0.0
        except:
            scores["context_inclusion"] = 0.0
    else:
        scores["context_inclusion"] = 1.0  # No skills to check

    # Criterion 4: Skill Relevance (only for NEW skills added by the model)
    provided_skills_str = ', '.join(scenario.context.skills) if scenario.context.skills else 'None'
    skill_relevance_prompt = f"""For a {scenario.context.job_title} position:

1. Extract ALL skills mentioned in the job offer (both in Required Skills and Nice-to-Have sections)
2. EXCLUDE these provided skills from evaluation (assume they are relevant): {provided_skills_str}
3. Evaluate ONLY the NEW skills added by the model for relevance to {scenario.context.job_title}
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

    skill_relevance_response = await get_judge_completion(skill_relevance_prompt, max_tokens=300)
    try:
        result = json.loads(clean_json_response(skill_relevance_response))
        scores["skill_relevance"] = result.get("score", 5) / 10
    except:
        scores["skill_relevance"] = 0.5

    # Criterion 5: Skill Completeness (check for missing obvious skills)
    completeness_prompt = f"""For a {scenario.context.job_title} position in {scenario.context.language} language:

Current skills in the job offer:
{generated_offer}

Are there any CRITICAL skills missing that are absolutely essential for this role?

Respond ONLY in JSON format:
{{
  "score": 0-10,
  "missing_critical_skills": []
}}"""

    completeness_response = await get_judge_completion(completeness_prompt, max_tokens=300)
    try:
        result = json.loads(clean_json_response(completeness_response))
        scores["skill_completeness"] = result.get("score", 5) / 10
    except:
        scores["skill_completeness"] = 0.5  # Default if parsing fails

    # Calculate final score (weighted average)
    final_score = (
        scores["language_consistency"] * 0.2 +  # 20%
        scores["xml_format"] * 0.2 +            # 20%
        scores["context_inclusion"] * 0.2 +     # 20%
        scores["skill_relevance"] * 0.2 +       # 20%
        scores["skill_completeness"] * 0.2      # 20%
    )

    # Update trajectory metrics
    trajectory.metrics.update(scores)
    trajectory.metrics["total_score"] = final_score
    trajectory.reward = final_score * 10  # Scale to 0-10

    # Debug output (occasional)
    if random.random() < 0.05:
        print("\n" + "="*50)
        print(f"Job Title: {scenario.context.job_title}")
        print(f"Language: {scenario.context.language}")
        print(f"Provided Skills: {scenario.context.skills}")
        print("-"*50)
        print("Scores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.2f}")
        print(f"Final Score: {final_score:.2f}")
        print("-"*50)
        print(f"Generated (first 500 chars):\n{generated_offer[:500]}...")
        print("="*50 + "\n")

    # Report to OpenPipe if configured
    if os.getenv("OPENPIPE_API_KEY"):
        try:
            op_client.report(
                requested_at=requested_at,
                received_at=int(time.time() * 1000),
                req_payload={
                    "model": model.name,
                    "messages": messages,
                    "metadata": {
                        "project": "job-offer-generation",
                        "step": scenario.step,
                        "language": scenario.context.language,
                        "job_title": scenario.context.job_title,
                        **scores,
                        "final_score": final_score,
                    },
                },
                resp_payload=completion,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return trajectory