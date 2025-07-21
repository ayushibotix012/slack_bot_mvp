# app/slack_listener.py

import os
import requests
from io import BytesIO
from docx import Document
from PIL import Image
import pytesseract
import fitz
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.utils.slack_utils import is_admin
from .openai_utils import ask_gpt
from dotenv import load_dotenv
from app.db.supabase_client import get_user_interactions, save_interaction
from slack_sdk.errors import SlackApiError
from .db.prompt_repo import update_system_prompt
from slack_bolt.context.respond import Respond
from slack_sdk.web import WebClient
from app.db.supabase_client import clear_user_interactions
from app.db.supabase_client import clear_all_interactions



load_dotenv()

slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Respond to a simple hello message
@slack_app.message("hello")
def handle_hello_message(message, say):
    user = message["user"]
    say(f"Hi <@{user}>! 👋 This is StakeholderBot.")

# 🔄 Handle file uploads sent via DM (file_share subtype in message event)
@slack_app.event("message")
def handle_user_message(body, client, logger):
    try:
        event = body.get("event", {})
        subtype = event.get("subtype", "")
        user_id = event.get("user")
        message_text = event.get("text", "").strip()
        files = event.get("files", [])

        if subtype == "bot_message":
            return  # Ignore messages sent by the bot

        # ✅ Step 1: Open a DM channel with the user
        dm = client.conversations_open(users=user_id)
        channel_id = dm["channel"]["id"]

        # ✅ Step 2: Send a temporary "thinking..." message
        thinking_msg = client.chat_postMessage(
            channel=channel_id,
            text="🤔 Thinking... please wait a moment."
        )
        thinking_ts = thinking_msg["ts"]

        # ✅ Step 3: Extract text from file if any
        text = ""
        if files:
            f = files[0]
            file_url = f.get("url_private_download")
            filetype = f.get("filetype")
            filename = f.get("name")

            headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}
            response = requests.get(file_url, headers=headers)

            if response.status_code != 200:
                client.chat_update(
                    channel=channel_id,
                    ts=thinking_ts,
                    text=f"❌ Failed to download `{filename}`."
                )
                return

            if filetype == "text":
                text = response.text
            elif filetype == "docx":
                doc = Document(BytesIO(response.content))
                text = "\n".join(p.text for p in doc.paragraphs)
            elif filetype in ["png", "jpg", "jpeg"]:
                image = Image.open(BytesIO(response.content))
                text = pytesseract.image_to_string(image)
            elif filetype == "pdf":
                pdf = fitz.open(stream=BytesIO(response.content), filetype="pdf")
                text = "".join([page.get_text() for page in pdf])
            else:
                client.chat_update(
                    channel=channel_id,
                    ts=thinking_ts,
                    text=f"⚠️ Unsupported file type: `{filetype}`"
                )
                return

        # ✅ Step 4: Ask GPT with the combined message + file text
        
        gpt_reply = ask_gpt(user_message=message_text, file_text=text,slack_user_id=user_id)

        # ✅ Step 5: Update the original message with GPT response and buttons
        client.chat_update(
            channel=channel_id,
            ts=thinking_ts,
            text=f"{gpt_reply[:1000]}",
            blocks=[
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{gpt_reply[:1000]}"}
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👍"},
                            "value": "thumbs_up",
                            "action_id": "feedback_like"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👎"},
                            "value": "thumbs_down",
                            "action_id": "feedback_dislike"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "❌"},
                            "value": "irrelevant",
                            "action_id": "feedback_error"
                        }
                    ]
                }
            ]
        )

        # ✅ Step 6: Get Slack user/org info
        user_info = client.users_info(user=user_id)
        user_name = user_info["user"]["real_name"]
        team_info = client.team_info()
        organization = team_info["team"]["name"]

        # ✅ Step 7: Save to Supabase
        save_interaction(
            slack_user_id=user_id,
            slack_user_name=user_name,
            organization=organization,
            message_text=message_text,
            extracted_text=text,
            response_text=gpt_reply,
            prompt_version="GPT-3.5",
            slack_ts=thinking_ts
        )

    except Exception as e:
        logger.error(f"❌ Error handling message: {e}")
        # Fallback error response
        try:
            client.chat_postMessage(channel=user_id, text="❌ Something went wrong while processing your message.")
        except Exception as inner:
            logger.error(f"❌ Failed to send fallback error message: {inner}")


@slack_app.command("/update")
def handle_update_prompt_command(ack, body, respond: Respond, client: WebClient):
    ack()

    user_id = body["user_id"]
    new_prompt = body["text"].strip()

    if not new_prompt:
        respond("⚠️ Please provide a new prompt.")
        return

    try:
        # Fetch user info from Slack
        user_info = client.users_info(user=user_id)
        is_admin = user_info["user"]["is_admin"]
        is_owner = user_info["user"].get("is_owner", False)

        if not (is_admin or is_owner):
            respond("❌ You are not authorized to update the system prompt.")
            return

    except SlackApiError as e:
        respond(f"❌ Failed to check user permissions: {e.response['error']}")
        return

    # Update the prompt in Supabase
    success = update_system_prompt(new_prompt, updated_by=user_id)

    if success:
        respond(f"✅ System prompt updated by <@{user_id}>.")
    else:
        respond("❌ Failed to update the system prompt. Please try again.")


@slack_app.action("feedback_like")
@slack_app.action("feedback_dislike")
@slack_app.action("feedback_error")
def handle_feedback(ack, body, action, client, logger):
    ack()  # Acknowledge the button click

    feedback_map = {
        "feedback_like": "👍",
        "feedback_dislike": "👎",
        "feedback_error": "❌"
    }

    feedback = feedback_map[action["action_id"]]
    user_id = body["user"]["id"]
    message_ts = body["message"]["ts"]

    try:
        # Update the record in Supabase using `message_ts` as a key
        from app.db.supabase_client import update_feedback
        update_feedback(message_ts, feedback)

        client.chat_postEphemeral(
            channel=body["channel"]["id"],
            user=user_id,
            text=f"Thank you for the feedback.: {feedback}"
        )
    except Exception as e:
        logger.error("Error updating feedback: %s", e)
        # client.chat_postEphemeral(
        #     channel=body["channel"]["id"],
        #     user=user_id,
        #     text="❌ Failed to save feedback."
        # )

@slack_app.command("/clear")
def handle_clear_command(ack, body, respond):
    ack()

    user_id = body["user_id"]
    success = clear_user_interactions(user_id)

    if success:
        respond(f"🧹 Your chat history has been cleared from memory.")
    else:
        respond(f"❌ Failed to clear your chat history. Please try again later.")


@slack_app.command("/clear_all")
def handle_clear_all_command(ack, body, client, respond):
    ack()
    user_id = body["user_id"]

    if not is_admin(user_id, client):
        respond("❌ You do not have permission to use this command.")
        return

    success = clear_all_interactions()

    if success:
        respond("🚨 All interactions have been permanently deleted from the database.")
    else:
        respond("❌ Failed to clear all interactions. Please try again later.")



# 🔁 Start the socket mode handler
def start_socket_mode():
    handler = SocketModeHandler(slack_app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
