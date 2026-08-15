import os
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

MODEL = "Qwen/Qwen3-8B"


def ask_llm(prompt: str, max_tokens: int = 500) -> str:

    for attempt in range(3):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens
            )

            message = response.choices[0].message

            content = message.content

            if content:
                return content.strip()

            # Qwen may return reasoning before final content
            reasoning_content = getattr(
                message,
                "reasoning_content",
                None
            )

            if reasoning_content:
                print(
                    "Warning: model returned reasoning "
                    "but no final answer."
                )

            print(
                f"Attempt {attempt + 1}: "
                "empty LLM response."
            )

        except Exception as e:

            print(
                f"LLM attempt {attempt + 1} failed: {e}"
            )

        if attempt < 2:
            time.sleep(2)

    return ""