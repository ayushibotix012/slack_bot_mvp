
import os
import base64
import mimetypes
# import imghdr
import filetype
import openai
import traceback
from dotenv import load_dotenv
from pathlib import Path
from io import BytesIO
from PIL import Image
import pytesseract  # OCR

from app.db.prompt_repo import get_system_prompt
from app.db.supabase_client import get_user_interactions
from app.vector_store_utils import query_vector_store

from openai import OpenAI

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
# openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))




def analyze_image_with_llm(img_bytes: bytes) -> str:
    """
    Analyze an image using GPT-4o and extract text + entities.
    Returns only the extracted response.
    """
    # # Detect image format
    # detected_format = imghdr.what(None, h=img_bytes)
    # mime_type = f"image/{detected_format}" if detected_format else "image/jpeg"
    kind = filetype.guess(img_bytes)
    mime_type = f"image/{kind.extension}" if kind else "image/jpeg"
    # Convert image to Base64
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    # System prompt
    system_prompt = (
        "You are an expert document/image analyzer.\n"
        "Your task:\n"
        "- Extract all text clearly from the image (like OCR).\n"
        "- Summarize key entities (names, dates, amounts, references).\n"
        "- If seals, signatures, or logos are present, describe them.\n"
        "- Maintain structure if possible (tables, sections).\n"
        "- If image is unclear, say 'Text partially unreadable'."
    )

    # User message with text + image
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please analyze this image and explain all of the information in detail."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
            ],
        },
    ]

    # Call GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.5
    )

    # Return only the text content
    return response.choices[0].message.content.strip()





# ----------------- QUERY CLASSIFIER -----------------
def llm_classify(user_message: str) -> str:
    """
    Use GPT for classification when heuristics are uncertain.
    """
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a classifier. Return only one word: general, document, image, or mixed."},
                {"role": "user", "content": user_message}
            ],
            temperature=0
        )
        label = resp.choices[0].message.content.strip().lower()
        if label in ["general", "document", "image", "mixed"]:
            return label
    except Exception:
        pass
    return "general"


def classify_query(user_message: str, file_text: str, images: list[bytes]) -> str:
    """
    Hybrid classifier: Heuristic first, GPT fallback for ambiguous queries.
    """
    text = user_message.lower().strip()

    # --- Explicit signals from inputs ---
    if images and file_text:
        return "mixed"
    if images:
        if any(word in text for word in ["see", "picture", "image", "photo", "screenshot", "visual"]):
            return "image"
        return "image"
    if file_text and file_text.strip():
        if any(word in text for word in ["document", "file", "pdf", "report", "text inside"]):
            return "document"
        return "document"

    # --- General by default ---
    label = "general"

    # If ambiguous phrasing, let GPT decide
    if any(word in text for word in ["this", "that", "above", "attached", "uploaded", "shown"]):
        label = llm_classify(user_message)

    return label


# --- Main GPT handler ---
def ask_gpt(user_message: str, file_text: str, slack_user_id: str) -> str:
    """
    Handles GPT response generation using user message, file text, and RAG context.
    """

    try:
        # --- Get system prompt ---
        base_system_prompt = get_system_prompt()

        # --- Fetch user conversation history ---
        past_interactions = get_user_interactions(slack_user_id)
        history_text = ""
        if past_interactions:
            for inter in past_interactions[-50:]:  # ✅ keep last 50 interactions
                extracted_part = f"\nExtracted Info: {inter['extracted_text']}" if inter.get("extracted_text") else ""
                history_text += (
                    f"User: {inter['message_text']}\n"
                    f"Assistant: {inter['response_text']}{extracted_part}\n"
                )

        # --- RAG Context ---
        rag_context = query_vector_store(user_message)

        # --- Build final extracted text ---
        combined_extracted_text = ""
        if file_text and file_text.strip():
            combined_extracted_text += file_text.strip()

        # --- System prompt with context ---
        # --- System prompt ---
        if rag_context.strip():
            system_prompt = f"""
        You are a professional and friendly AI assistant. Use all available context to answer queries accurately. 
        - Use the vector store context when relevant.
        - Focus on document or file content if the query relates to a file.
        - Combine conversation history and context intelligently.
        - Always be polite, professional, and clear.
        - Avoid mentioning internal system details.
        Vector Store Context:
        {rag_context}
        """
        else:
            system_prompt = f"""
        You are a professional and friendly AI assistant. 
        - Answer general queries using your knowledge and conversation history.
        - Provide polite and friendly greetings when appropriate.
        - Keep answers clear, accurate, and professional.
        - Avoid referring to vector stores or documents unless they exist.
        """


        # --- Final user prompt to GPT ---
        final_prompt = f"""
        Conversation History:
        {history_text}

        User Query:
        {user_message}

        Extracted Text:
        {combined_extracted_text}
        """


        # --- GPT API Call (new SDK style) ---
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.4,
            max_tokens=1000,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        traceback.print_exc()
        return f"⚠️ Error in ask_gpt: {str(e)}"



