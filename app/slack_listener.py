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
from .openai_utils import ask_gpt, analyze_image_with_llm
from dotenv import load_dotenv
from app.db.supabase_client import get_user_interactions, save_interaction
from slack_sdk.errors import SlackApiError
from .db.prompt_repo import update_system_prompt
from slack_bolt.context.respond import Respond
from slack_sdk.web import WebClient
from app.db.supabase_client import clear_user_interactions
from app.db.supabase_client import clear_all_interactions

# ------------------ LangChain RAG implementation ----------------
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------------- #
# Vector store
from app.vector_store_utils import add_to_vector_store, clear_vector_store, load_vector_store

load_dotenv()

slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Tokenizer-aware chunking function
def split_text_into_chunks(text, chunk_size=3000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# ----------------- SLACK LISTENER -----------------
@slack_app.event("message")
def handle_user_message(body, client, logger):
    try:
        load_vector_store()
        event = body.get("event", {})
        subtype = event.get("subtype", "")
        user_id = event.get("user")
        raw_text = event.get("text", "")
        message_text = raw_text.strip() if isinstance(raw_text, str) else ""
        files = event.get("files", [])
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts", event.get("ts"))

        if subtype == "bot_message":
            return

        thinking_msg = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text="🤔 Analyzing your message and files..."
        )
        thinking_ts = thinking_msg["ts"]

        extracted_texts = []
        headers = {"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"}

        for f in files or []:
            try:
                file_url = f.get("url_private_download")
                filetype = (f.get("filetype") or "").lower()
                mimetype = (f.get("mimetype") or "").lower()
                filename = f.get("name")
                logger.info(f"📥 Downloading: {filename}")
                resp = requests.get(file_url, headers=headers)
                if resp.status_code != 200:
                    logger.error(f"❌ Failed to download {filename} (status {resp.status_code})")
                    continue

                is_image = filetype in ["png", "jpg", "jpeg"] or mimetype.startswith("image/")

                if filetype == "text" or mimetype in ["text/plain"]:
                    extracted_texts.append(resp.text)

                elif filetype == "docx" or mimetype in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    doc = Document(BytesIO(resp.content))
                    extracted_texts.append("\n".join(p.text for p in doc.paragraphs))

                elif filetype == "pdf" or mimetype == "application/pdf":
                    import fitz
                    pdf = fitz.open(stream=BytesIO(resp.content), filetype="pdf")
                    extracted_texts.append("".join([page.get_text() for page in pdf]))

                elif is_image:
                    # --- GPT Vision analysis ---
                    print(resp.content)
                    gpt_image_text = analyze_image_with_llm(resp.content)
                    print(gpt_image_text)
                    print("*************---------------------*************")
                    extracted_texts.append(gpt_image_text)

                    user_info = client.users_info(user=user_id)
                    team_info = client.team_info()
                    save_interaction(
                        slack_user_id=user_id,
                        slack_user_name=user_info["user"]["real_name"],
                        organization=team_info["team"]["name"],
                        message_text=gpt_image_text,
                        extracted_text=None,
                        response_text=None,
                        prompt_version="RAG-GPT4",
                        slack_ts=thinking_ts
                    )
                else:
                    logger.warning(f"⚠️ Unsupported file type: {filetype or mimetype}")
                    extracted_texts.append(f"[Unsupported file type: {filetype or mimetype}]")

            except Exception as file_err:
                logger.error(f"❌ Error processing {filename}: {file_err}")

        # Combine extracted text
        extracted_combined_text = "\n\n".join([t for t in extracted_texts if isinstance(t, str)])

        # Add to vector store if we have text
        if extracted_combined_text.strip():
            chunks = split_text_into_chunks(extracted_combined_text, chunk_size=5000, chunk_overlap=300)
            add_to_vector_store(chunks)

        # Ask GPT
        final_response = ask_gpt(
            user_message=message_text,
            file_text=extracted_combined_text,
            slack_user_id=user_id
        )

        # Update Slack message
        client.chat_update(
            channel=channel_id,
            ts=thinking_ts,
            text=final_response,
            blocks=[
                {"type": "section", "text": {"type": "mrkdwn", "text": final_response}},
                {"type": "actions", "elements": [
                    {"type": "button", "text": {"type": "plain_text", "text": "👍"}, "value": "thumbs_up", "action_id": "feedback_like"},
                    {"type": "button", "text": {"type": "plain_text", "text": "👎"}, "value": "thumbs_down", "action_id": "feedback_dislike"},
                    {"type": "button", "text": {"type": "plain_text", "text": "❌"}, "value": "irrelevant", "action_id": "feedback_error"}
                ]}
            ]
        )

        # Save full interaction
        user_info = client.users_info(user=user_id)
        team_info = client.team_info()
        save_interaction(
            slack_user_id=user_id,
            slack_user_name=user_info["user"]["real_name"],
            organization=team_info["team"]["name"],
            message_text=message_text,
            extracted_text=extracted_combined_text,
            response_text=final_response,
            prompt_version="RAG-GPT4",
            slack_ts=thinking_ts
        )

    except Exception as e:
        logger.error(f"❌ Error handling message: {e}")
        try:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="❌ Something went wrong while processing your message."
            )
        except Exception as inner:
            logger.error(f"❌ Failed to send fallback message: {inner}")



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


@slack_app.command("/clear")
def handle_clear_command(ack, body, respond):
    ack()

    user_id = body["user_id"]
    success = clear_user_interactions(user_id)
    clearVector = clear_vector_store()

    if success and clearVector:
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
