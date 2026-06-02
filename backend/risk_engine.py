import os
from typing import Any

def _weak_topic_count(weak_topics):
    return len(weak_topics or [])

def _score_trend(quiz_attempts):
    scores = [float(a.percentage or 0) for a in quiz_attempts][-5:]
    if len(scores) < 3:
        return "stable"
    first = sum(scores[:len(scores)//2]) / len(scores[:len(scores)//2])
    second = sum(scores[len(scores)//2:]) / len(scores[len(scores)//2:])
    if second >= first + 5:
        return "improving"
    if second <= first - 5:
        return "declining"
    return "stable"

def extract_risk_features(profile, quiz_attempts, exam_marks) -> dict[str, Any]:
    exam_percentages = [float(m.percentage or 0) for m in exam_marks]
    failed_exam_count = sum(1 for p in exam_percentages if p < 50)

    return {
        "avg_score": float(profile.avg_score or 0),
        "weak_topic_count": _weak_topic_count(profile.weak_topics),
        "days_inactive": int(profile.days_inactive or 0),
        "failed_exam_count": failed_exam_count,
        "avg_exam_percentage": (
            round(sum(exam_percentages) / len(exam_percentages), 1)
            if exam_percentages else 0
        ),
        "score_trend": _score_trend(quiz_attempts),
    }

def calculate_risk_score(features: dict[str, Any]) -> int:
    score = 0

    avg_score = features["avg_score"]
    if avg_score < 40:
        score += 35
    elif avg_score < 60:
        score += 25
    elif avg_score < 75:
        score += 10

    weak_count = features["weak_topic_count"]
    if weak_count >= 5:
        score += 25
    elif weak_count >= 3:
        score += 15
    elif weak_count >= 1:
        score += 8

    inactive = features["days_inactive"]
    if inactive >= 14:
        score += 20
    elif inactive >= 7:
        score += 12
    elif inactive >= 3:
        score += 5

    failed = features["failed_exam_count"]
    if failed >= 3:
        score += 20
    elif failed >= 1:
        score += 10

    exam_avg = features["avg_exam_percentage"]
    if exam_avg and exam_avg < 50:
        score += 15
    elif exam_avg and exam_avg < 65:
        score += 8

    if features["score_trend"] == "declining":
        score += 15
    elif features["score_trend"] == "improving":
        score -= 10

    return max(0, min(100, round(score)))

def classify_risk_level(risk_score: int) -> str:
    if risk_score >= 70:
        return "High"
    if risk_score >= 40:
        return "Medium"
    return "Low"

def generate_risk_reasons(features: dict[str, Any]) -> list[str]:
    reasons = []

    if features["avg_score"] < 60:
        reasons.append("Low average score")
    if features["weak_topic_count"] >= 3:
        reasons.append("Many weak topics")
    elif features["weak_topic_count"] > 0:
        reasons.append("Some weak topics need practice")
    if features["days_inactive"] >= 7:
        reasons.append("Inactive for several days")
    if features["failed_exam_count"] > 0:
        reasons.append("Failed recent exam marks")
    if features["avg_exam_percentage"] and features["avg_exam_percentage"] < 60:
        reasons.append("Low exam average")
    if features["score_trend"] == "declining":
        reasons.append("Declining performance trend")

    return reasons or ["No major risk signals detected"]

def compute_student_risk(profile, quiz_attempts, exam_marks) -> dict[str, Any]:
    features = extract_risk_features(profile, quiz_attempts, exam_marks)
    risk_score = calculate_risk_score(features)

    return {
        "risk_score": risk_score,
        "risk_level": classify_risk_level(risk_score),
        "risk_reasons": generate_risk_reasons(features),
        "risk_features": features,
    }

def generate_intervention_recommendation(student, profile) -> str:
    try:
        from langchain_groq import ChatGroq
    except Exception:
        return "Review weak topics, assign remedial quizzes, and schedule a follow-up practice session."

    weak_topics = profile.weak_topics or []
    weak_text = ", ".join(
        f"{w.get('topic')} ({w.get('subject')})" if isinstance(w, dict) else str(w)
        for w in weak_topics
    ) or "No weak topics recorded"

    prompt = f"""
Suggest one concise intervention for this student.

Student: {student.name}
Risk score: {profile.risk_score}
Risk level: {profile.risk_level}
Weak topics: {weak_text}
Risk reasons: {", ".join(profile.risk_reasons or [])}

Return one practical recommendation in 1-2 sentences.
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )
    return llm.invoke(prompt).content