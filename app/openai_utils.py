import os
import base64
import mimetypes
import imghdr
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

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(user_message: str, file_text: str, slack_user_id: str, images: list[bytes] = None) -> str:
    """
    Ask GPT with text + optional image(s) + RAG context.
    If images are provided, OCR will be performed to extract text automatically.
    """
    base_system_prompt = get_system_prompt()
    history = get_user_interactions(slack_user_id=slack_user_id, limit=50)

    try:
        rag_context = query_vector_store(user_message)
    except Exception as e:
        rag_context = ""
        print("⚠️ Vector store query failed:", e)

    # --- Auto OCR if images provided ---
    extracted_ocr_text = ""
    if images:
        for img_bytes in images:
            try:
                img = Image.open(BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    extracted_ocr_text += ocr_text.strip() + "\n"
            except Exception as e:
                print(f"⚠️ OCR failed for one image: {e}")

    # Prefer file_text, but append OCR text if found
    combined_extracted_text = ""
    if file_text and file_text.strip():
        combined_extracted_text += file_text.strip()
    if extracted_ocr_text.strip():
        combined_extracted_text += "\n\n[OCR Extracted Text]\n" + extracted_ocr_text.strip()

    system_prompt = f"""{base_system_prompt}

You may receive both text and images from the user.
When images are provided, examine them visually in detail **and** consider the OCR extracted text.
Combine visual observations with any provided OCR text or vector store context when relevant.

Vector Store Context:
{rag_context}
"""

    # Build conversation
    messages = [{"role": "system", "content": system_prompt}]

    for interaction in reversed(history):
        messages.append({"role": "user", "content": interaction["message_text"]})
        messages.append({"role": "assistant", "content": interaction["response_text"]})

    # Build latest user content
    latest_user_content = [{"type": "text", "text": user_message}]

    if combined_extracted_text.strip():
        latest_user_content.append({
            "type": "text",
            "text": f"\n\nBelow is the extracted text from the uploaded file/image:\n{combined_extracted_text}"
        })
    elif images:
        latest_user_content.append({
            "type": "text",
            "text": "\n\nNo extracted text available — analyze the image visually."
        })

    # Attach images if any
    if images:
        for img_bytes in images:
            detected_format = imghdr.what(None, h=img_bytes)
            mime_type = f"image/{detected_format}" if detected_format else "image/jpeg"
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            latest_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            })

    messages.append({"role": "user", "content": latest_user_content})

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # multimodal
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ OpenAI API Error:")
        traceback.print_exc()
        return "❌ Failed to get a response from GPT."
