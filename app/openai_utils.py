import os
import openai
from dotenv import load_dotenv
from pathlib import Path
from .db.prompt_repo import get_system_prompt
from .db.supabase_client import get_user_interactions  # ✅ Import the history fetch function

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(user_message: str, file_text: str, slack_user_id: str) -> str:
    system_prompt = get_system_prompt()

    # ✅ Load up to 5 past messages from this user
    history = get_user_interactions(slack_user_id=slack_user_id, limit=20)

    # ✅ Start building the message list with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # ✅ Add previous conversation history (oldest to latest)
    for interaction in reversed(history):
        messages.append({"role": "user", "content": interaction["message_text"]})
        messages.append({"role": "assistant", "content": interaction["response_text"]})

    # ✅ Add the current message (plus any file content)
    combined_message = f"""User message:
{user_message}

File content:
{file_text[:3000]}""" if file_text else user_message

    messages.append({"role": "user", "content": combined_message})

    # ✅ Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "❌ Failed to get a response from GPT."
