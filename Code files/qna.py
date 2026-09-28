"""
qna.py - Question & Answer module

Uses Google Gemini to answer general knowledge /
academic questions with concise, accurate responses.
"""

import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def answer_question_with_gemini(question: str) -> str:
    """Send a question to Gemini and return a concise answer."""
    if not GEMINI_API_KEY:
        return "⚠️ Error in QnA: GEMINI_API_KEY is not set. Please add it to your .env file."

    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")
        response = model.generate_content(question)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in QnA: {e}"