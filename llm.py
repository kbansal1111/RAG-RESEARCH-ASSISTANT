from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_answer(question, context):

    prompt = f"""
You are a helpful research assistant.

Use ONLY the provided context.

If the answer is not present in the context, say:
'I could not find that information in the provided documents.'

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content