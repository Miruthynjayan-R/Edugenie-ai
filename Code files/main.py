"""
main.py - EduGenie FastAPI application

Wires together the frontend (HTML/CSS via Jinja2 + static files) and
the five AI modules (Q&A, Explanation, Summarization, Quiz Generation,
Learning Path) behind a small REST API.
"""

import os
from dotenv import load_dotenv
print("Loaded key starts with:", os.getenv("GEMINI_API_KEY")[:6] if os.getenv("GEMINI_API_KEY") else "NOT FOUND")
# Load environment variables (GEMINI_API_KEY) from .env before anything
# that reads them at import time.
load_dotenv()

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from qna import answer_question_with_gemini
from explanation_module import explain_topic
from summary_module import summarize_text
from quiz_module import generate_quiz
from learning_path import get_learning_recommendations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="EduGenie - Google Gemini Powered Learning Assistant")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/")
async def home(request: Request):
    """Serve the EduGenie web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------
# Q&A - GET API
# ---------------------------------------------------------------------
@app.get("/qa")
async def answer_question(question: str = Query(...)):
    answer = answer_question_with_gemini(question)
    return {"answer": answer}


# ---------------------------------------------------------------------
# Explanation - POST API
# ---------------------------------------------------------------------
@app.post("/explain/")
async def explain_api(request: Request):
    data = await request.json()
    topic = data.get("topic")
    if not topic:
        return JSONResponse(content={"error": "Please provide a topic."}, status_code=400)
    explanation = explain_topic(topic)
    return {"topic": topic, "explanation": explanation}


# ---------------------------------------------------------------------
# Summarization - POST API
# ---------------------------------------------------------------------
@app.post("/summarize/")
async def summarize_api(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return JSONResponse(content={"error": "Please provide text to summarize."}, status_code=400)
    summary = summarize_text(text)
    return {"summary": summary}


# ---------------------------------------------------------------------
# Quiz Generation - POST API
# ---------------------------------------------------------------------
@app.post("/quiz")
async def quiz_api(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return JSONResponse(content={"error": "Please provide text for quiz."}, status_code=400)
    quiz = generate_quiz(text)
    return JSONResponse(content={"quiz": quiz})


# ---------------------------------------------------------------------
# Learning Recommendations - GET API
# ---------------------------------------------------------------------
@app.get("/learn/recommendations")
async def learning_recommendation_api(topic: str = Query(...)):
    recommendation = get_learning_recommendations(topic)
    return {"topic": topic, "recommendation": recommendation}


# ---------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EduGenie"}