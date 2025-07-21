import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}


def get_system_prompt():
    """Fetch the latest system prompt"""
    url = f"{SUPABASE_URL}/rest/v1/system_prompt?select=prompt,updated_by,updated_at&order=updated_at.desc&limit=1"
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200 and response.json():
        prompt_data = response.json()[0]
        return prompt_data["prompt"]
    else:
        print("❌ Failed to fetch system prompt:", response.text)
        return "You are a helpful assistant."  # fallback default


def update_system_prompt(new_prompt: str, updated_by: str):
    """Update the system prompt"""
    url = f"{SUPABASE_URL}/rest/v1/system_prompt"
    payload = {
        "prompt": new_prompt,
        "updated_by": updated_by,
        "updated_at": datetime.utcnow().isoformat()
    }

    response = requests.post(url, json=payload, headers=HEADERS)

    if response.status_code in [200, 201]:
        print("✅ System prompt updated.")
        return True
    else:
        print("❌ Failed to update prompt:", response.status_code, response.text)
        return False
