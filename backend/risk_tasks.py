import models
from database import SessionLocal
from risk_engine import compute_student_risk

def recompute_student_risk(student_id: int, db):
    profile = db.query(models.StudentProfile).filter_by(user_id=student_id).first()
    if not profile:
        return None

    quiz_attempts = (
        db.query(models.QuizAttempt)
        .filter_by(student_id=student_id)
        .order_by(models.QuizAttempt.timestamp.asc())
        .all()
    )
    exam_marks = db.query(models.ExamMark).filter_by(student_id=student_id).all()

    risk = compute_student_risk(profile, quiz_attempts, exam_marks)
    profile.risk_score = risk["risk_score"]
    profile.risk_level = risk["risk_level"]
    profile.risk_reasons = risk["risk_reasons"]
    return risk

def recompute_all_student_risks():
    db = SessionLocal()
    try:
        students = db.query(models.User).filter_by(role="student").all()
        results = []
        for student in students:
            risk = recompute_student_risk(student.id, db)
            if risk:
                results.append({"student_id": student.id, **risk})
        db.commit()
        return results
    finally:
        db.close()