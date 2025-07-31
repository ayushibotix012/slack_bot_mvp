import tiktoken
from app.openai_utils import ask_gpt  # ✅ Ensure this is your custom function

# Choose appropriate tokenizer for your model
encoding = tiktoken.encoding_for_model("gpt-4")  # or "gpt-3.5-turbo"

def split_text_into_chunks(text: str, max_tokens: int = 6000) -> list[str]:
    """
    Splits the input text into chunks, each within the token limit.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        token_count = len(encoding.encode(word))
        if current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = token_count
        else:
            current_chunk.append(word)
            current_tokens += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_document_in_chunks(document_text: str, user_query: str, slack_user_id: str) -> str:
    """
    Splits the document into chunks, runs ask_gpt() on each chunk, then summarizes all answers.
    """
    print("📄 Splitting document into chunks...")
    chunks = split_text_into_chunks(document_text)

    chunk_answers = []
    for i, chunk in enumerate(chunks):
        print(f"🤖 Processing chunk {i+1}/{len(chunks)}...")
        response = ask_gpt(user_message=user_query, file_text=chunk, slack_user_id=slack_user_id)
        chunk_answers.append(f"Answer from chunk {i+1}:\n{response}")

    combined_answers = "\n\n".join(chunk_answers)

    print("🧠 Sending combined answers for final summary...")
    final_summary = ask_gpt(
        user_message=f"""Here are the answers from different parts of a document based on the same question: "{user_query}".
Please summarize or provide a final unified answer from all the responses.""",
        file_text=combined_answers,
        slack_user_id=slack_user_id
    )

    return final_summary
