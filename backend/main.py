from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from database import get_db, engine
from datetime import datetime
from typing import Optional
import models, auth
import shutil, os, json, csv, io
from risk_tasks import recompute_student_risk, recompute_all_student_risks
from risk_engine import generate_intervention_recommendation
from fastapi.responses import StreamingResponse
import asyncio

from agent_handler import (
    handle_chat,
    handle_quiz_start,
    handle_quiz_submit,
    get_or_create_session,
    close_session,
    build_quiz_context
)
CURRICULUM_FILE = "curriculum.json"
# Import rag for PDF upload
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_helper import get_rag


# Create all tables on startup
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth helpers ─────────────────────────────────────────────

def get_current_user(
    authorization: str = None,
    db: Session = Depends(get_db)
):
    from fastapi import Header
    return authorization


from fastapi import Header

def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    payload = auth.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(models.User).filter_by(id=int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_professor(
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != "professor":
        raise HTTPException(status_code=403, detail="Professor access only")
    return current_user


# ── Pydantic schemas ─────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str = "student"

class LoginRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    topic: str = "general"
    session_id: Optional[int] = None

class QuizStartRequest(BaseModel):
    topic: str
    difficulty: str = "medium"
    num_questions: int = 5
    message: str = ""
    session_id: Optional[int] = None

class QuizAnswerRequest(BaseModel):
    thread_id: str
    answers: list
    topic: str
    difficulty: str = "medium"
    session_id: Optional[int] = None

class ProfileUpdate(BaseModel):
    level: str
    learning_style: str = "mixed"

class ExamMarkInput(BaseModel):
    student_email: str
    exam_name: str
    subject: str
    marks_obtained: float
    total_marks: float
    conducted_on: str
#from rag_evaluator import run_rag_evaluation
from rag_helper import RAGHelper
from rag_helper import get_rag

class RAGEvalRequest(BaseModel):
    questions: list  # list of {"question": str, "ground_truth": str}

@app.post("/professor/evaluate-rag")
async def evaluate_rag(
    req: RAGEvalRequest,
    prof: models.User = Depends(require_professor)
):
    try:
        from rag_evaluator import run_rag_evaluation
    except Exception as e:
        return {
        "error": f"RAG evaluator unavailable: {str(e)}"
    }
    from graphs.chat_graph import chat_flow
    test_cases = []

    for item in req.questions:
        question = item["question"]
        # Retrieve chunks
        chunks = get_rag().query(question, k=4)
        contexts = [
            c.page_content if hasattr(c, "page_content") else str(c)
            for c in chunks
        ]
        # Get LLM answer
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        answer = llm.invoke(
            f"Answer this question using the context:\n"
            f"Context: {chr(10).join(contexts)}\n"
            f"Question: {question}"
        ).content

        test_cases.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item.get("ground_truth", answer)
        })

    scores = run_rag_evaluation(test_cases)
    return scores

# ═══════════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter_by(email=req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=req.email,
        hashed_password=auth.hash_password(req.password),
        name=req.name,
        role=req.role
    )
    db.add(user)
    db.flush()

    if req.role == "student":
        profile = models.StudentProfile(user_id=user.id)
        db.add(profile)

    db.commit()
    return {"message": "Registered successfully"}


@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=req.email).first()
    if not user or not auth.verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last_active and login streak
    if user.role == "student":
        profile = db.query(models.StudentProfile).filter_by(user_id=user.id).first()
        if profile:
            profile.last_active = datetime.utcnow()
            profile.days_inactive = 0
    db.commit()

    token = auth.create_token({"sub": str(user.id), "role": user.role})
    return {"token": token, "role": user.role, "name": user.name, "id": user.id}

@app.post("/upload-curriculum")
async def upload_curriculum(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != "professor":
        raise HTTPException(
            status_code=403,
            detail="Only professors can upload curriculum"
        )

    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files allowed"
        )

    content = (await file.read()).decode("utf-8")

    curriculum = []
    blocks = content.strip().split("\n\n")

    for block in blocks:

        lines = [l.strip() for l in block.splitlines() if l.strip()]

        subject = None
        topics = []

        for line in lines:

            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()

            elif line.startswith("Topics:"):
                topics = [
                    t.strip()
                    for t in line.replace("Topics:", "").split(",")
                    if t.strip()
                ]

        if subject:
            curriculum.append({
                "subject": subject,
                "topics": topics
            })

    with open(CURRICULUM_FILE, "w") as f:
        json.dump(curriculum, f, indent=2)

    return {
        "message": "Curriculum uploaded successfully",
        "subjects": len(curriculum)
    }

@app.get("/curriculum")
def get_curriculum(
    current_user: models.User = Depends(get_current_user)
):
    if not os.path.exists(CURRICULUM_FILE):
        return {"curriculum": []}

    with open(CURRICULUM_FILE, "r") as f:
        curriculum = json.load(f)

    return {
        "curriculum": curriculum
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════
# SESSION ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/session/start")
def start_session(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = get_or_create_session(current_user.id, db)
    return {"session_id": session.id, "started_at": session.started_at}


@app.post("/session/end/{session_id}")
def end_session(
    session_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    result = close_session(session_id, current_user.id, db)
    return {"message": "Session ended", "analysis": result}


@app.get("/session/active")
def get_active_session(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(models.StudentSession)\
        .filter_by(student_id=current_user.id, is_active=True)\
        .first()
    if not session:
        return {"session_id": None}
    return {"session_id": session.id, "started_at": session.started_at}


# ═══════════════════════════════════════════════════════════════
# CHAT ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/chat/message")
def chat_message(
    req: ChatRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not req.session_id:
        session = get_or_create_session(current_user.id, db)
        session_id = session.id
    else:
        session_id = req.session_id

    result = handle_chat(
        message=req.message,
        topic=req.topic,
        student_id=current_user.id,
        session_id=session_id,
        db=db
    )
    return {
        "reply": result["response"],
        "sources": result.get("sources", []),
        "session_id": session_id
    }

@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not req.session_id:
        session = get_or_create_session(current_user.id, db)
        session_id = session.id
    else:
        session_id = req.session_id

    async def event_generator():
        full_response = ""

        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.4,
                streaming=True
            )

            prompt = f"""
You are Lumina, an AI tutor.

Student question:
{req.message}

Topic:
{req.topic}

Respond clearly and helpfully.
"""

            # Save user message
            db.add(models.ChatMessage(
                student_id=current_user.id,
                session_id=session_id,
                role="user",
                content=req.message
            ))
            db.commit()

            for chunk in llm.stream(prompt):
                token = chunk.content or ""
                if not token:
                    continue

                full_response += token
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

                await asyncio.sleep(0)

            # Save assistant response after stream finishes
            db.add(models.ChatMessage(
                student_id=current_user.id,
                session_id=session_id,
                role="assistant",
                content=full_response
            ))
            db.commit()

            yield f"data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/chat/sessions")
def get_chat_sessions(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(models.StudentSession)\
        .filter_by(student_id=current_user.id)\
        .order_by(models.StudentSession.started_at.desc())\
        .all()

    result = []
    for s in sessions:
        msg_count  = db.query(models.ChatMessage).filter_by(session_id=s.id).count()
        quiz_count = db.query(models.QuizAttempt).filter_by(session_id=s.id).count()

        # Title = first user message truncated
        first_msg = db.query(models.ChatMessage)\
            .filter_by(session_id=s.id, role="user")\
            .order_by(models.ChatMessage.timestamp.asc())\
            .first()
        title = (first_msg.content[:40] + "...") if first_msg else "New session"

        result.append({
            "session_id": s.id,
            "title": title,
            "started_at": s.started_at.strftime("%d %b, %I:%M %p"),
            "ended_at": s.ended_at.strftime("%d %b, %I:%M %p") if s.ended_at else None,
            "is_active": s.is_active,
            "message_count": msg_count,
            "quiz_count": quiz_count,
            "summary": s.summary
        })
    return result


@app.get("/chat/session/{session_id}")
def get_session_messages(
    session_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    messages = db.query(models.ChatMessage)\
        .filter_by(session_id=session_id, student_id=current_user.id)\
        .order_by(models.ChatMessage.timestamp.asc())\
        .all()

    return [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp.strftime("%I:%M %p")
        }
        for m in messages
    ]


# ═══════════════════════════════════════════════════════════════
# QUIZ ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/quiz/start")
def quiz_start(
    req: QuizStartRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not req.session_id:
        session = get_or_create_session(current_user.id, db)
        session_id = session.id
    else:
        session_id = req.session_id

    result = handle_quiz_start(
        topic=req.topic,
        student_id=current_user.id,
        session_id=session_id,
        difficulty=req.difficulty,
        num_questions=req.num_questions,
        message=req.message,
        db=db
    )
    return result


@app.post("/quiz/answer")
def quiz_answer(
    req: QuizAnswerRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not req.session_id:
        session = get_or_create_session(current_user.id, db)
        session_id = session.id
    else:
        session_id = req.session_id

    result = handle_quiz_submit(
        thread_id=req.thread_id,
        answers=req.answers,
        topic=req.topic,
        difficulty=req.difficulty,
        student_id=current_user.id,
        session_id=session_id,
        db=db
    )
    return result


@app.get("/quiz/history")
def quiz_history(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    attempts = db.query(models.QuizAttempt)\
        .filter_by(student_id=current_user.id)\
        .order_by(models.QuizAttempt.timestamp.desc())\
        .all()

    return [
        {
            "id": a.id,
            "topic": a.topic,
            "difficulty": a.difficulty,
            "score": a.score,
            "total": a.total,
            "percentage": a.percentage,
            "feedback": a.feedback,
            "timestamp": a.timestamp.strftime("%d %b %Y, %I:%M %p")
        }
        for a in attempts
    ]


@app.get("/quiz/history/{attempt_id}")
def quiz_history_detail(
    attempt_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    attempt = db.query(models.QuizAttempt)\
        .filter_by(id=attempt_id, student_id=current_user.id)\
        .first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Quiz attempt not found")

    return {
        "id": attempt.id,
        "topic": attempt.topic,
        "difficulty": attempt.difficulty,
        "score": attempt.score,
        "total": attempt.total,
        "percentage": attempt.percentage,
        "feedback": attempt.feedback,
        "questions": attempt.questions_json,
        "user_answers": attempt.user_answers_json,
        "timestamp": attempt.timestamp.strftime("%d %b %Y, %I:%M %p")
    }


# ═══════════════════════════════════════════════════════════════
# STUDENT ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/student/profile")
def get_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "avg_score": profile.avg_score,
        "weak_topics": profile.weak_topics or [],
        "risk_score": profile.risk_score,
        "level": profile.level,
        "learning_style": profile.learning_style,
        "recommended_topic": profile.recommended_topic,
        "confidence_level": profile.confidence_level,
        "risk_level": profile.risk_level or "Low",
        "risk_reasons": profile.risk_reasons or []
    }

@app.get("/student/profile/full")
def get_full_profile(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=current_user.id).first()
    return {
        "name": current_user.name,
        "email": current_user.email,
        "level": profile.level or "beginner",
        "learning_style": profile.learning_style or "mixed",
        "avg_score": profile.avg_score or 0,
        "weak_topics": profile.weak_topics or [],
        "risk_score": profile.risk_score or 0,
        "attempt_count": profile.attempt_count or 0,
        "recommended_topic": profile.recommended_topic,
        "confidence_level": profile.confidence_level,
        "risk_level": profile.risk_level or "Low",
        "risk_reasons": profile.risk_reasons or []
    }


@app.put("/student/profile/update")
def update_profile(
    req: ProfileUpdate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=current_user.id).first()
    profile.level = req.level
    profile.learning_style = req.learning_style
    db.commit()
    return {"message": "Profile updated"}


@app.get("/student/progress")
def get_progress(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=current_user.id).first()
    attempts = db.query(models.QuizAttempt)\
        .filter_by(student_id=current_user.id)\
        .order_by(models.QuizAttempt.timestamp.asc()).all()

    topic_scores = {}
    for a in attempts:
        topic_scores.setdefault(a.topic, []).append(a.percentage)

    topic_avg = {
        t: round(sum(s)/len(s), 1)
        for t, s in topic_scores.items()
    }

    return {
        "avg_score": profile.avg_score or 0,
        "total_attempts": len(attempts),
        "weak_topics": profile.weak_topics or [],
        "risk_score": profile.risk_score or 0,
        "level": profile.level or "beginner",
        "recommended_topic": profile.recommended_topic,
        "confidence_level": profile.confidence_level,
        "topic_avg": topic_avg,
        "risk_level": profile.risk_level or "Low",
        "risk_reasons": profile.risk_reasons or [],
        "recent_attempts": [
            {
                "topic": a.topic,
                "score": a.percentage,
                "timestamp": a.timestamp.strftime("%d %b, %I:%M %p")
            }
            for a in reversed(attempts[-10:])
        ],
        "score_history": [
            {"attempt": i+1, "score": a.percentage, "topic": a.topic}
            for i, a in enumerate(attempts)
        ]
    }


@app.get("/student/exam-marks")
def student_exam_marks(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    marks = db.query(models.ExamMark)\
        .filter_by(student_id=current_user.id)\
        .order_by(models.ExamMark.conducted_on.desc())\
        .all()
    return [
        {
            "exam_name": m.exam_name,
            "subject": m.subject,
            "marks_obtained": m.marks_obtained,
            "total_marks": m.total_marks,
            "percentage": m.percentage,
            "grade": m.grade,
            "conducted_on": m.conducted_on.strftime("%d %b %Y")
        }
        for m in marks
    ]


# ═══════════════════════════════════════════════════════════════
# PDF UPLOAD
# ═══════════════════════════════════════════════════════════════

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != "professor":
        raise HTTPException(status_code=403, detail="Only professors can upload course material")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    success = rag.load_pdf(path)
    return {"message": f"'{file.filename}' uploaded and embedded" if success else "Upload failed"}

@app.get("/uploaded-pdfs")
async def list_pdfs(current_user: models.User = Depends(get_current_user)):
    os.makedirs("uploads", exist_ok=True)
    files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]
    try:
        count = rag.get_document_count()
    except:
        count = 0
    return {"files": files, "chunks_in_db": count}


# ═══════════════════════════════════════════════════════════════
# PROFESSOR ROUTES
# ═══════════════════════════════════════════════════════════════

def calculate_grade(percentage: float) -> str:
    if percentage >= 90: return "A+"
    elif percentage >= 80: return "A"
    elif percentage >= 70: return "B"
    elif percentage >= 60: return "C"
    elif percentage >= 50: return "D"
    else: return "F"


@app.post("/professor/upload-mark")
def upload_single_mark(
    req: ExamMarkInput,
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    student = db.query(models.User)\
        .filter_by(email=req.student_email, role="student").first()
    if not student:
        raise HTTPException(
            status_code=404, detail=f"Student '{req.student_email}' not found"
        )
    percentage = round((req.marks_obtained / req.total_marks) * 100, 1)
    grade = calculate_grade(percentage)
    mark = models.ExamMark(
        student_id=student.id,
        professor_id=prof.id,
        exam_name=req.exam_name,
        subject=req.subject,
        marks_obtained=req.marks_obtained,
        total_marks=req.total_marks,
        percentage=percentage,
        grade=grade,
        conducted_on=datetime.strptime(req.conducted_on, "%Y-%m-%d")
    )
    db.add(mark)
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=student.id).first()
    if profile and percentage < 50:
        weak = set(profile.weak_topics or [])
        weak.add(req.subject.lower())
        profile.weak_topics = list(weak)

    recompute_student_risk(student.id, db)
    db.commit()
    return {"message": f"Mark uploaded for {student.name}",
            "grade": grade, "percentage": percentage}


@app.post("/professor/upload-marks-csv")
async def upload_marks_csv(
    file: UploadFile = File(...),
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    success, errors = [], []
    for i, row in enumerate(reader, start=2):
        try:
            email = row["email"].strip()
            student = db.query(models.User)\
                .filter_by(email=email, role="student").first()
            if not student:
                errors.append(f"Row {i}: Student '{email}' not found")
                continue
            marks = float(row["marks_obtained"])
            total = float(row["total_marks"])
            percentage = round((marks / total) * 100, 1)
            grade = calculate_grade(percentage)
            mark = models.ExamMark(
                student_id=student.id,
                professor_id=prof.id,
                exam_name=row["exam_name"].strip(),
                subject=row["subject"].strip(),
                marks_obtained=marks,
                total_marks=total,
                percentage=percentage,
                grade=grade,
                conducted_on=datetime.strptime(
                    row["conducted_on"].strip(), "%Y-%m-%d"
                )
            )
            db.add(mark)
            profile = db.query(models.StudentProfile)\
                .filter_by(user_id=student.id).first()
            if profile and percentage < 50:
                weak = set(profile.weak_topics or [])
                weak.add(row["subject"].strip().lower())
                profile.weak_topics = list(weak)
            success.append(student.name)
        except Exception as e:
            errors.append(f"Row {i}: {str(e)}")

    recompute_student_risk(student.id, db)
    db.commit()
    return {"uploaded": len(success), "errors": errors, "students": success}


@app.get("/professor/overview")
def professor_overview(
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    students = db.query(models.User).filter_by(role="student").all()
    profiles = db.query(models.StudentProfile).all()
    at_risk = [p for p in profiles if (p.risk_score or 0) > 70]
    avg_score = (
        sum(p.avg_score or 0 for p in profiles) / len(profiles)
        if profiles else 0
    )
    all_weak = []
    for p in profiles:
        all_weak.extend(p.weak_topics or [])
    topic_freq = {}

    for t in all_weak:
        if isinstance(t, dict):
            key = f"{t['subject']}::{t['topic']}"
        else:
            key = t

    topic_freq[key] = topic_freq.get(key, 0) + 1
    top_weak = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return {
        "total_students": len(students),
        "at_risk_count": len(at_risk),
        "class_avg_score": round(avg_score, 1),
        "top_weak_topics": [
    {
        "topic": t.split("::")[1] if "::" in t else t,
        "subject": t.split("::")[0] if "::" in t else "",
        "count": c
    }
    for t, c in top_weak
]
    }


@app.get("/professor/students")
def professor_students(
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    students = db.query(models.User).filter_by(role="student").all()
    result = []
    for s in students:
        profile = db.query(models.StudentProfile)\
            .filter_by(user_id=s.id).first()
        attempts = db.query(models.QuizAttempt)\
            .filter_by(student_id=s.id).count()
        result.append({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "level": profile.level if profile else "beginner",
            "avg_score": round(profile.avg_score or 0, 1) if profile else 0,
            "risk_score": round(profile.risk_score or 0, 1) if profile else 0,
            "weak_topics": profile.weak_topics or [] if profile else [],
            "attempt_count": attempts,
            "last_active": profile.last_active.strftime("%d %b %Y")
                if profile and profile.last_active else "Never"
        })
    result.sort(key=lambda x: x["risk_score"], reverse=True)
    return result


@app.get("/professor/student/{student_id}")
def professor_student_detail(
    student_id: int,
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    student = db.query(models.User)\
        .filter_by(id=student_id, role="student").first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    profile = db.query(models.StudentProfile)\
        .filter_by(user_id=student_id).first()
    attempts = db.query(models.QuizAttempt)\
        .filter_by(student_id=student_id)\
        .order_by(models.QuizAttempt.timestamp.asc()).all()
    topic_scores = {}
    for a in attempts:
        topic_scores.setdefault(a.topic, []).append(a.percentage)
    topic_avg = {
        t: round(sum(s) / len(s), 1) for t, s in topic_scores.items()
    }
    return {
        "name": student.name,
        "email": student.email,
        "level": profile.level if profile else "beginner",
        "avg_score": round(profile.avg_score or 0, 1) if profile else 0,
        "risk_score": round(profile.risk_score or 0, 1) if profile else 0,
        "weak_topics": profile.weak_topics or [] if profile else [],
        # "understood_topics": profile.understood_topics or [] if profile else [],
        "attempt_count": len(attempts),
        "topic_avg": topic_avg,
        "risk_level": profile.risk_level or "Low",
        "risk_reasons": profile.risk_reasons or [],
        "intervention_recommendation": generate_intervention_recommendation(student, profile),
        "score_history": [
            {"attempt": i + 1, "score": a.percentage, "topic": a.topic}
            for i, a in enumerate(attempts)
        ],
        "last_active": profile.last_active.strftime("%d %b %Y, %I:%M %p")
            if profile and profile.last_active else "Never"
    }


@app.get("/professor/at-risk")
def professor_at_risk(
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    profiles = db.query(models.StudentProfile)\
        .filter(models.StudentProfile.risk_score > 70)\
        .order_by(models.StudentProfile.risk_score.desc()).all()
    result = []
    for p in profiles:
        student = db.query(models.User).filter_by(id=p.user_id).first()
        if student:
            result.append({
                "name": student.name,
                "email": student.email,
                "risk_score": round(p.risk_score, 1),
                "weak_topics": p.weak_topics or [],
                "avg_score": round(p.avg_score or 0, 1),
                "days_inactive": p.days_inactive or 0
            })
    return result


@app.get("/professor/exam-analytics")
def exam_analytics(
    prof: models.User = Depends(require_professor),
    db: Session = Depends(get_db)
):
    marks = db.query(models.ExamMark).filter_by(professor_id=prof.id).all()
    if not marks:
        return {
            "exams": [], "subject_avg": {},
            "grade_dist": {}, "total_entries": 0
        }
    exam_groups = {}
    for m in marks:
        key = f"{m.exam_name} — {m.subject}"
        exam_groups.setdefault(key, []).append(m.percentage)
    exams = [
        {
            "name": key,
            "avg": round(sum(v) / len(v), 1),
            "highest": max(v),
            "lowest": min(v),
            "count": len(v)
        }
        for key, v in exam_groups.items()
    ]
    subject_groups = {}
    for m in marks:
        subject_groups.setdefault(m.subject, []).append(m.percentage)
    subject_avg = {
        s: round(sum(v) / len(v), 1) for s, v in subject_groups.items()
    }
    grade_dist = {}
    for m in marks:
        grade_dist[m.grade] = grade_dist.get(m.grade, 0) + 1
    return {
        "total_entries": len(marks),
        "exams": exams,
        "subject_avg": subject_avg,
        "grade_dist": grade_dist
    }

@app.post("/professor/recompute-risks")
def recompute_risks(
    prof: models.User = Depends(require_professor)
):
    return {"updated": recompute_all_student_risks()}