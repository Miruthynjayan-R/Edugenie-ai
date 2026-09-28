"""
summary_module.py - Summarization module

Uses Google Gemini's generative capabilities to condense long
passages into clear, concise summaries while retaining key
information.
"""

import os
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def summarize_text(text: str) -> str:
    """Summarize `text` into a concise, easy-to-understand version."""
    if not GEMINI_API_KEY:
        return "⚠️ Error in Summary: GEMINI_API_KEY is not set. Please add it to your .env file."

    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")
        prompt = f"Summarize the following text in simple language:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error in Summary: {e}"