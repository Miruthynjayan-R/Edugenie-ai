"""
quiz_module.py - Quiz Generation module

Uses Google Gemini to generate 3 multiple-choice questions (MCQs)
from a given passage/topic, each with 4 options and a correct answer,
returned as structured JSON.
"""

import os
import re
import json
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def clean_json_block(text: str) -> str:
    """Strip Markdown code fences (```json ... ```) if present."""
    return re.sub(r"```(?:json)?\n(.*?)```", r"\1", text, flags=re.DOTALL).strip()


def generate_quiz(text: str) -> list:
    """Generate 3 MCQs (4 options each) from the given passage/topic."""
    if not GEMINI_API_KEY:
        return [{"error": "GEMINI_API_KEY is not set. Please add it to your .env file."}]

    try:
        model = genai.GenerativeModel(model_name="models/gemini-3.8-flash")

        prompt = f"""
You are a quiz generator.

From the following passage, create 3 multiple-choice questions. Each question should include:
- A "question"
- A list of 4 "options"
- A correct "answer" that must exactly match one of the options.

Format your output as **valid JSON**, like this:
[
  {{
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }}
]

Passage:
{text}
"""
        response = model.generate_content(prompt)
        quiz_text = response.text.strip()

        # Clean markdown code blocks if any
        cleaned_text = clean_json_block(quiz_text)

        quiz_data = json.loads(cleaned_text)
        return quiz_data
    except json.JSONDecodeError as e:
        return [{"error": f"Could not parse quiz JSON from model response: {e}"}]
    except Exception as e:
        return [{"error": f"Error generating quiz: {e}"}]