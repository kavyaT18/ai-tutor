from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Optional, List
from database import SessionLocal
import models, os
from dotenv import load_dotenv
load_dotenv()


class MatchResult(BaseModel):
    matched_topic: Optional[str] = Field(None)
    subject: Optional[str] = Field(None)


import json

CURRICULUM_FILE = "curriculum.json"

def get_curriculum() -> List[dict]:
    if not os.path.exists(CURRICULUM_FILE):
        return []

    try:
        with open(CURRICULUM_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def match_topic_to_curriculum(detected_topic: str) -> Optional[dict]:
    """
    Returns {"topic": ..., "subject": ...} if match found, else None.
    """
    curriculum = get_curriculum()
    if not curriculum:
        return None

    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    structured = model.with_structured_output(MatchResult)

    curriculum_text = "\n".join(
        f"Subject: {e['subject']}\nTopics: {', '.join(e['topics'])}"
        for e in curriculum
    )

    prompt = f"""You are a curriculum matcher.

Curriculum:
{curriculum_text}

Detected topic from student activity: "{detected_topic}"

Task: Find the closest matching topic from the curriculum above.
If nothing is reasonably close, return null for both fields.
Do not force a match if it does not fit."""

    try:
        result = structured.invoke(prompt)
        if result.matched_topic and result.subject:
            return {"topic": result.matched_topic, "subject": result.subject}
        return None
    except Exception:
        return None


def match_topics_list(topics: List[str]) -> List[dict]:
    """Match a list of topics, return only successfully matched ones."""
    matched = []
    for t in topics:
        result = match_topic_to_curriculum(t)
        if result:
            matched.append(result)
    return matched