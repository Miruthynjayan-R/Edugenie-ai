"""
learning_path.py - Personalized Learning Path module

Uses Google Gemini to generate a structured, adaptive learning path
(beginner -> intermediate -> advanced) for any topic the learner
wants to explore, including suggested resources.
"""

import os
import traceback
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_learning_recommendations(topic: str) -> str:
    """Generate a structured learning path for the given topic."""
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY is not set. Please add it to your .env file."

    prompt = f"""
You are an AI tutor. The student wants to learn about: {topic}.
Suggest a structured and adaptive learning path including key topics, order of learning, and resources (books, videos, articles).
Include beginner, intermediate, and advanced levels if needed.
"""
    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")
        response = model.generate_content(prompt)

        if hasattr(response, "text") and response.text:
            return response.text
        elif hasattr(response, "parts") and response.parts:
            return response.parts[0].text
        else:
            return "❌ Could not extract content from Gemini response."
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error occurred: {str(e)}"