# app/db/supabase_client.py

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def save_interaction(
    slack_user_id: str,
    slack_user_name: str,
    organization: str,
    message_text: str,
    extracted_text: str,
    response_text: str,
    prompt_version: str = "GPT-3.5",  # Default version for prompt
    feedback: str = None, # Optional; e.g., 👍, 👎, ❌
    slack_ts: str = None
):
    data = {
        "slack_user_id": slack_user_id,
        "slack_user_name": slack_user_name,
        "organization": organization,
        "message_text": message_text,
        "extracted_text": extracted_text,
        "response_text": response_text,
        "prompt_version": prompt_version,
        "feedback": feedback,
        "slack_ts": slack_ts
    }

    url = f"{SUPABASE_URL}/rest/v1/interactions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 201:
        print("✅ Interaction saved to Supabase.")
    else:
        print("❌ Failed to save interaction:", response.status_code)
        print(response.text)


def update_feedback(message_ts, feedback):
    url = f"{SUPABASE_URL}/rest/v1/interactions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    params = {"slack_ts": f"eq.{message_ts}"}
    data = {"feedback": feedback}

    response = requests.patch(url, params=params, json=data, headers=headers)

    if response.status_code == 204:
        print("✅ Feedback updated.")
    else:
        print("❌ Failed to update feedback:", response.status_code, response.text)


def get_user_interactions(slack_user_id: str, limit: int = 50):
    url = f"{SUPABASE_URL}/rest/v1/interactions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    params = {
        "slack_user_id": f"eq.{slack_user_id}",
        "order": "created_at.desc",
        "limit": str(limit)
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()  # List of previous messages
    else:
        print("❌ Failed to fetch interactions:", response.status_code)
        return []


# app/db/supabase_client.py

def clear_user_interactions(slack_user_id: str):
    url = f"{SUPABASE_URL}/rest/v1/interactions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    params = {"slack_user_id": f"eq.{slack_user_id}"}

    response = requests.delete(url, headers=headers, params=params)

    if response.status_code == 204:
        print(f"✅ Cleared history for user: {slack_user_id}")
        return True
    else:
        print("❌ Failed to clear user history:", response.status_code, response.text)
        return False


# app/db/supabase_client.py

def clear_all_interactions():
    url = f"{SUPABASE_URL}/rest/v1/interactions"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.delete(url, headers=headers)

    if response.status_code == 204:
        print("✅ All interactions cleared.")
        return True
    else:
        print("❌ Failed to clear all interactions:", response.status_code, response.text)
        return False


def save_image_context(conversation_id: str, image_id: str, extracted_data: dict):
    """
    Save extracted image data (OCR + description + purpose) to Supabase.
    """
    url = f"{SUPABASE_URL}/rest/v1/image_contexts"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "conversation_id": conversation_id,
        "image_id": image_id,
        "extracted_data": json.dumps(extracted_data)  # stored as JSON in Supabase
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 201:
        print("✅ Image context saved to Supabase.")
        return response.json()
    else:
        print("❌ Failed to save image context:", response.status_code, response.text)
        return None
    

def get_image_context(conversation_id: str, image_id: str = None):
    """
    Fetch previously saved image context for a conversation (and optionally a specific image).
    """
    url = f"{SUPABASE_URL}/rest/v1/image_contexts"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    params = {"conversation_id": f"eq.{conversation_id}"}
    if image_id:
        params["image_id"] = f"eq.{image_id}"

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        print("❌ Failed to fetch image context:", resp.status_code, resp.text)
        return []    