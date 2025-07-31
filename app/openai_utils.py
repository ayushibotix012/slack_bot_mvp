import os
import openai
import traceback
from dotenv import load_dotenv
from pathlib import Path

from app.db.prompt_repo import get_system_prompt
from app.db.supabase_client import get_user_interactions

from app.vector_store_utils import query_vector_store

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

import tiktoken

def num_tokens_from_string(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def trim_text_to_token_limit(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    trimmed = tokens[:max_tokens]
    return encoding.decode(trimmed)

def ask_gpt(user_message: str, file_text: str, slack_user_id: str) -> str:
    base_system_prompt = get_system_prompt()
    history = get_user_interactions(slack_user_id=slack_user_id, limit=50)

    try:
        rag_context = query_vector_store(user_message)
    except Exception as e:
        rag_context = ""
        print("⚠️ Vector store query failed:", e)

    # ✅ Truncate to safe limits
    rag_context = trim_text_to_token_limit(rag_context, 2000)
    file_text = trim_text_to_token_limit(file_text, 2000)

    system_prompt = f"""{base_system_prompt}

You have access to relevant content retrieved from the document vector store.
Use this context only if it's helpful to the user's message.

Vector Store Context:
{rag_context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Only add history if total token count permits
    total_tokens = num_tokens_from_string(system_prompt + user_message + file_text)
    for interaction in reversed(history):
        if total_tokens >= 7000:
            break
        user_q = interaction["message_text"]
        assistant_a = interaction["response_text"]
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a})
        total_tokens += num_tokens_from_string(user_q + assistant_a)

    # Final user message
    if file_text:
        final_user_prompt = f"""{user_message}

Below is the text extracted from the uploaded file (if relevant):
{file_text}
"""
    else:
        final_user_prompt = user_message

    messages.append({"role": "user", "content": final_user_prompt})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ OpenAI API Error:")
        traceback.print_exc()
        return "❌ Failed to get a response from GPT."

