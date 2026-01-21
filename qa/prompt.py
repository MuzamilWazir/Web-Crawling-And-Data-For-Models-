# qa/prompt.py
def build_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant.
Answer ONLY from the website content below.
If the answer is not present, say: "Information not found on this website."

Website Content:
{context}

Question:
{question}
"""
