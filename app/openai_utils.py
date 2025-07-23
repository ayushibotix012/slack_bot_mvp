import os
import openai
from dotenv import load_dotenv
from pathlib import Path
from .db.prompt_repo import get_system_prompt
from .db.supabase_client import get_user_interactions  # ✅ Get past chats from DB

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt(user_message: str, file_text: str, slack_user_id: str) -> str:
    # ✅ Load system prompt from DB
    base_system_prompt = get_system_prompt()

    # ✅ Load past messages + extracted file texts from DB
    history = get_user_interactions(slack_user_id=slack_user_id, limit=20)

    # ✅ Combine extracted file content from past interactions
    extracted_blobs = [item["extracted_text"] for item in history if item.get("extracted_text")]
    extracted_context = "\n\n".join(extracted_blobs)

    # ✅ Final system prompt including extracted context
    system_prompt = f"""{base_system_prompt}

Relevant extracted context from previous documents or images (if applicable):
{extracted_context[:3000]}
"""

    # ✅ Instruction for GPT on how to treat file content
    instruction = """The user may provide a message along with optional file content (e.g., a document, image, or PDF). 
Only use the file content if the user's message clearly refers to or depends on the file. 
If the message does not mention or relate to the file, completely ignore the file and respond based only on the message.
Be precise in your judgment. If the file appears unrelated to the question, treat it as irrelevant."""

    # ✅ Start building the GPT message list
    messages = [{"role": "system", "content": system_prompt}]

    # ✅ Add previous chat history (user + assistant) in order
    for interaction in reversed(history):
        messages.append({"role": "user", "content": interaction["message_text"]})
        messages.append({"role": "assistant", "content": interaction["response_text"]})

    # ✅ Combine current message and file content (if any)
    if file_text:
        combined_message = f"""User message: {user_message} + {instruction}

File content:
{file_text[:3000]}
"""
    else:
        combined_message = user_message

    # ✅ Append current query to the message list
    messages.append({"role": "user", "content": combined_message})

    # ✅ Send to OpenAI
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
