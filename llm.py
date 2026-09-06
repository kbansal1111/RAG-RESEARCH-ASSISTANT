from openai import OpenAI
from config import Config
from retry_handler import RetryHandler
from monitor import monitor_latency, logger
import time

client = OpenAI(
    api_key=Config.OPENROUTER_API_KEY,
    base_url=Config.OPENROUTER_BASE_URL
)

class LLMGenerator:
    """LLM answer generation with retry logic and monitoring"""
    
    def __init__(self, monitor=None):
        self.monitor = monitor
        
    @RetryHandler.retry_with_backoff()
    @monitor_latency('llm_generation')
    def generate_answer(self, question, context):
        """Generate answer using LLM with retry logic"""
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
            model=Config.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content