from graphs.quiz_graph import start_quiz, submit_answers
from graphs.chat_graph import send_message
from graphs.analyzer_graph import run_analyzer
from sqlalchemy.orm import Session
from datetime import datetime
import models
from risk_tasks import recompute_student_risk


# ── Quiz context builder ─────────────────────────────────────
# Used by chat to know what student has been tested on

def build_quiz_context(student_id: int, db: Session, limit: int = 5) -> str:
    attempts = db.query(models.QuizAttempt)\
        .filter_by(student_id=student_id)\
        .order_by(models.QuizAttempt.timestamp.desc())\
        .limit(limit).all()

    if not attempts:
        return ""

    lines = ["Recent quiz performance:"]
    for a in reversed(attempts):
        lines.append(
            f"- {a.topic} ({a.difficulty}): "
            f"{a.score}/{a.total} = {a.percentage:.0f}%"
        )
    return "\n".join(lines)


# ── Session management ───────────────────────────────────────

def get_or_create_session(student_id: int, db: Session) -> models.StudentSession:
    """Return active session or create a new one."""
    session = (
        db.query(models.StudentSession)
        .filter_by(student_id=student_id, is_active=True)
        .first()
    )
    if not session:
        session = models.StudentSession(student_id=student_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


from topic_matcher import match_topics_list

def close_session(session_id: int, student_id: int, db: Session) -> dict:
    session = db.query(models.StudentSession).filter_by(id=session_id).first()
    if not session:
        return {"error": "Session not found"}

    chat_messages = db.query(models.ChatMessage)\
        .filter_by(session_id=session_id)\
        .order_by(models.ChatMessage.timestamp.asc()).all()
    quiz_attempts = db.query(models.QuizAttempt)\
        .filter_by(session_id=session_id).all()

    student = db.query(models.User).filter_by(id=student_id).first()
    profile = db.query(models.StudentProfile).filter_by(user_id=student_id).first()

    analysis = run_analyzer(
        student_name=student.name,
        current_level=profile.level or "beginner",
        current_weak_topics=[
            w["topic"] if isinstance(w, dict) else w
            for w in (profile.weak_topics or [])
        ],
        chat_messages=chat_messages,
        quiz_attempts=quiz_attempts
    )

    if analysis:
        # Match weak topics through curriculum
        raw_weak = analysis.get("weak_topics", [])
        matched_weak = match_topics_list(raw_weak)

        # Merge with existing — avoid duplicates by topic name
        existing = {
            w["topic"].lower(): w
            for w in (profile.weak_topics or [])
            if isinstance(w, dict)
        }
        for w in matched_weak:
            existing[w["topic"].lower()] = w

        profile.weak_topics = list(existing.values())

        if analysis.get("learning_style"):
            profile.learning_style = analysis["learning_style"]
        if analysis.get("level"):
            profile.level = analysis["level"]
        if analysis.get("confidence_level"):
            profile.confidence_level = analysis["confidence_level"]
        if analysis.get("recommended_topic"):
            profile.recommended_topic = analysis["recommended_topic"]

        profile.last_active = datetime.utcnow()

    session.is_active = False
    session.ended_at = datetime.utcnow()
    session.summary = analysis
    db.commit()
    return analysis


# ── Chat handler ─────────────────────────────────────────────

def handle_chat(
    message: str,
    topic: str,
    student_id: int,
    session_id: int,
    db: Session
) -> dict:
    from graphs.chat_graph import send_message
    thread_id = f"chat_{student_id}_{session_id}"
    result = send_message(
        thread_id=thread_id,
        message=message,
        student_id=student_id
    )
    return result

# ── Quiz handler ─────────────────────────────────────────────

def handle_quiz_start(
    topic: str,
    student_id: int,
    session_id: int,
    difficulty: str,
    num_questions: int,
    message: str,
    db: Session
) -> dict:
    """Start quiz graph, return questions to frontend."""

    profile = db.query(models.StudentProfile).filter_by(user_id=student_id).first()
    thread_id = f"quiz_{student_id}_{session_id}_{topic.replace(' ', '_')}"

    result = start_quiz(
        thread_id=thread_id,
        topic=topic,
        knowledge_level=profile.level or "beginner",
        learning_style=profile.learning_style or "mixed",
        difficulty_level=difficulty,
        num_questions=num_questions,
        message=message
    )

    # Store thread_id so /quiz/answer can resume it
    result["thread_id"] = thread_id
    result["session_id"] = session_id
    return result


def handle_quiz_submit(
    thread_id: str,
    answers: list,
    topic: str,
    difficulty: str,
    student_id: int,
    session_id: int,
    db: Session
) -> dict:
    """Resume quiz graph with answers, save result to DB."""

    result = submit_answers(thread_id=thread_id, answers=answers)

    # Save to DB
    attempt = models.QuizAttempt(
        student_id=student_id,
        session_id=session_id,
        topic=topic,
        difficulty=difficulty,
        num_questions=result["total"],
        score=result["score"],
        total=result["total"],
        percentage=result["percentage"],
        questions_json=result["questions"],
        user_answers_json=result["user_answers"],
        feedback=result["feedback"]
    )
    db.add(attempt)

    # Update avg_score and attempt_count on profile
    profile = db.query(models.StudentProfile).filter_by(user_id=student_id).first()
    all_attempts = (
        db.query(models.QuizAttempt)
        .filter_by(student_id=student_id)
        .all()
    )
    profile.attempt_count = len(all_attempts) + 1
    total_score = sum(a.percentage for a in all_attempts) + result["percentage"]
    profile.avg_score = round(total_score / profile.attempt_count, 1)

    # Immediately update weak topics from this quiz
    # Immediately update weak topics from this quiz

    raw_weak = result.get("weak_topics", [])
    matched_weak = match_topics_list(raw_weak)

    existing = {}

# Existing weak topics
    for w in (profile.weak_topics or []):
        if isinstance(w, dict):
            existing[f"{w['subject']}::{w['topic']}"] = w

# New weak topics from this quiz
    for w in matched_weak:
        if isinstance(w, dict):
            existing[f"{w['subject']}::{w['topic']}"] = w



    

    profile.weak_topics = list(existing.values())  
    recompute_student_risk(student_id, db)
  

    db.commit()

    result["attempt_id"] = attempt.id
    return result