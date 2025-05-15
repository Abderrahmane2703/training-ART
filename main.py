from datasets import load_dataset
import art
from art.local import LocalAPI
import asyncio
from dotenv import load_dotenv
import csv
import random
import re
from typing import TypedDict, List, Tuple, Dict, Optional, Union
from openai import AsyncOpenAI
import json
import glob
import os
import regex

ds = load_dataset("ServiceNow/repliqa")

OPENROUTER_API_KEY = ""
MODEL = "google/gemini-2.5-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

import time
import requests
from requests.exceptions import RequestException, Timeout

semaphore = asyncio.Semaphore(20)

from async_lru import alru_cache

@alru_cache(maxsize=1024)
async def chat_completion(prompt, temperature=0.0, max_tokens=600, retries=3, timeout=10):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                response = await asyncio.to_thread(
                    requests.post,
                    OPENROUTER_URL,
                    headers=HEADERS,
                    json=payload,
                    timeout=timeout
                )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries:
                print(f"[Retry {attempt}/{retries}] chat_completion failed: {e}. Retrying...")
                await asyncio.sleep(3)
            else:
                print(f"[Failure] chat_completion failed after {retries} attempts: {e}")
                return "ERROR: Chat completion failed"



load_dotenv()

model = art.TrainableModel(
    name="014",
    project="summarize",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
)

async def rollout(
    client: AsyncOpenAI, doc: Dict, use_full=False
) -> art.Trajectory:
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content":
                f"""
You are a specialized AI assistant that generates concise, informative summaries for documents.
"""
            }
        ],
        reward=0,
        metrics={"word_count": 0, "len": 0, "percent": 0, "percent_full": 0, "percent_diff": 0}
    )

    summarize_prompt = f"""You are a specialized AI assistant that generates concise, informative summaries for documents.

Here is a document: {doc['document']}

Generate a summary that conveys all relevant information in a concise manner."""

    trajectory.messages_and_choices.append(
        {
            "role": "user",
            "content": summarize_prompt
        }
    )

    messages = trajectory.messages()
    chat = await client.chat.completions.create(
        messages=messages,
        model=model.name,
        max_tokens=1000
    )
    choice = chat.choices[0]
    if use_full:
        choice.message.content = doc['document']
    trajectory.messages_and_choices.append(choice)
    summary = choice.message.content

    total_score = 0
    total_score_full = 0
    total_questions = 0
    
    for question in doc['questions']:
        total_questions += 1
        if not regex.search(r'\p{Han}', summary) and len(summary) <= 3000:
            prompt = f"Here is a document: {summary}\n\nAnswer this question to the best of your ability in one sentence, if the document does not contain the answer, just state so: {question['q']}"
            response = await chat_completion(prompt)

            judge_prompt = f"Here is a document: {doc['document']}\n\nHere is a question: {question['q']}\n\nHere is a generated answer: {response}\n\nHere is the golden answer: {question['a']}\n\nIf the answers mostly match return a 1, if they do not match return a 0. Do not return any other text."

            score = await chat_completion(judge_prompt)
            try:
                total_score += int(score)
            except:
                pass

        prompt_full = f"Here is a document: {doc['document']}\n\nAnswer this question to the best of your ability in one sentence, if the document does not contain the answer, just state so: {question['q']}"
        response_full = await chat_completion(prompt_full)

        judge_prompt_full = f"Here is a document: {doc['document']}\n\nHere is a question: {question['q']}\n\nHere is a generated answer: {response_full}\n\nHere is the golden answer: {question['a']}\n\nIf the answers mostly match return a 1, if they do not match return a 0. Do not return any other text."

        score_full = await chat_completion(judge_prompt_full)
        try:
            total_score_full += int(score_full)
        except:
            pass

        if not regex.search(r'\p{Han}', summary) and len(summary) <= 3000:
            if random.random() < 0.05:
                print("Answers:")
                print(summary)
                print("question", question['q'])
                print("golden:", question['a'])
                print("generated:", response)
                print("score:", score)
                print("score-full:", score_full)
                print("generated-full:", response_full)
                print("\n\n\n\n\n")
    
    trajectory.metrics['percent'] = total_score / total_questions
    trajectory.metrics['percent_full'] = total_score_full / total_questions
    trajectory.metrics['percent_diff'] = trajectory.metrics['percent'] - trajectory.metrics['percent_full']
    trajectory.metrics['word_count'] = len(summary.split())
    trajectory.metrics['len'] = len(summary)
    trajectory.reward = total_score

    return trajectory


async def main():
    ds = load_dataset("ServiceNow/repliqa")
    documents = {}

    for data in ds["repliqa_0"]:
        if data['document_id'] not in documents:
            documents[data['document_id']] = {'document': data['document_extracted'], 'questions': []}
        documents[data['document_id']]['questions'].append({'q': data['question'], 'a': data['answer']})

    all_documents = []
    for doc_id in documents:
        all_documents.append(documents[doc_id])

    random.seed(80)
    random.shuffle(all_documents)

    # Now split into train/val/test sets
    val_size = min(25, len(all_documents) // 10)

    val_puzzles = all_documents[:val_size]
    train_puzzles = all_documents[val_size:]

    await model.register(LocalAPI())

    batch_size = 10  # Process this many puzzles per batch
    num_epochs = 1  # Number of complete passes through the training data
    openai_client = model.openai_client()
    
    start_step = await model.get_step()
    max_steps = 1000
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_puzzles)
        
        # Calculate how many batches we can process in this epoch
        num_batches = min(len(train_puzzles) // batch_size, (max_steps - start_step) // num_epochs)
        
        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break
                
            print(f"Epoch {epoch+1}, Batch {batch+1}/{num_batches}, Step {current_step}")
            
            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size
            
            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for t in range(2))
                        for puzzle in val_puzzles
                    ),
                    pbar_desc=f"val (epoch {epoch+1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for t in range(10))
                        for puzzle in train_puzzles[batch_start_idx:batch_end_idx]
                    ),
                    pbar_desc=f"train (epoch {epoch+1}, batch {batch+1})",
                ),
            )

            await model.log(val_groups)
            await model.delete_checkpoints()
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )

if __name__ == "__main__":
    asyncio.run(main())
