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


# ----------------- MAIN GPT HANDLER -----------------
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

    # --- Classify query type ---
    query_type = classify_query(user_message, file_text, images)
    print(f"📝 Query classified as: {query_type}")

    # --- Build system prompt with classification ---
    if rag_context.strip():
        system_prompt = f"""{base_system_prompt}

Query Type: {query_type}

- If relevant information is found in the **Vector Store Context**, use it to answer the question accurately.
- If the context is empty or unrelated, respond using your own knowledge.
- For greetings or casual conversation (e.g. "hi", "hello", "how are you"), reply naturally and politely.
- If the query type is 'document', focus on the extracted file text.
- If the query type is 'image', use both OCR and visual analysis.
- If the query type is 'mixed', combine all sources intelligently.
- Always give a clear, helpful answer without referencing system details.

Vector Store Context:
{rag_context}
"""
    else:
        system_prompt = f"""{base_system_prompt}

Query Type: {query_type}

- Answer questions using your own knowledge unless a file or image is provided.
- For greetings or casual conversation (e.g. "hi", "hello", "how are you"), reply naturally and politely.
- If the query type is 'document', focus on the extracted file text.
- If the query type is 'image', use both OCR and visual analysis.
- If the query type is 'mixed', combine all sources intelligently.
- Always give a clear, helpful answer without referencing system details.
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
