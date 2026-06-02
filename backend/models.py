from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id                = Column(Integer, primary_key=True, index=True)
    email             = Column(String, unique=True, index=True)
    hashed_password   = Column(String)
    name              = Column(String)
    role              = Column(String, default="student")
    created_at        = Column(DateTime, default=datetime.utcnow)

    profile           = relationship("StudentProfile", back_populates="user", uselist=False)
    quiz_attempts     = relationship("QuizAttempt", back_populates="student", foreign_keys="QuizAttempt.student_id")
    chat_messages     = relationship("ChatMessage", back_populates="student", foreign_keys="ChatMessage.student_id")
    sessions          = relationship("StudentSession", back_populates="student")
    exam_marks        = relationship("ExamMark", back_populates="student", foreign_keys="ExamMark.student_id")


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id                   = Column(Integer, primary_key=True, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), unique=True)
    level                = Column(String, default="beginner")
    learning_style       = Column(String, default="mixed")
    weak_topics          = Column(JSON, default=list)
    #understood_topics    = Column(JSON, default=list)
    avg_score            = Column(Float, default=0.0)
    risk_score           = Column(Float, default=0.0)
    risk_level = Column(String, default="Low")
    risk_reasons = Column(JSON, default=list) 
    login_streak         = Column(Integer, default=0)
    days_inactive        = Column(Integer, default=0)
    attempt_count        = Column(Integer, default=0)
    last_active          = Column(DateTime, default=datetime.utcnow)
    recommended_topic    = Column(String, nullable=True)   # set by analyzer
    confidence_level     = Column(String, default="unknown")  # set by analyzer

    user = relationship("User", back_populates="profile")


class StudentSession(Base):
    """One session = one study sitting. Groups chat + quiz together."""
    __tablename__ = "student_sessions"

    id           = Column(Integer, primary_key=True, index=True)
    student_id   = Column(Integer, ForeignKey("users.id"))
    started_at   = Column(DateTime, default=datetime.utcnow)
    ended_at     = Column(DateTime, nullable=True)
    is_active    = Column(Boolean, default=True)
    summary      = Column(JSON, nullable=True)   # analyzer output

    student       = relationship("User", back_populates="sessions")
    chat_messages = relationship("ChatMessage", back_populates="session")
    quiz_attempts = relationship("QuizAttempt", back_populates="session")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id                 = Column(Integer, primary_key=True, index=True)
    student_id         = Column(Integer, ForeignKey("users.id"))
    session_id         = Column(Integer, ForeignKey("student_sessions.id"), nullable=True)
    topic              = Column(String)
    difficulty         = Column(String, default="medium")
    num_questions      = Column(Integer, default=5)
    score              = Column(Integer, default=0)       # raw correct count
    total              = Column(Integer, default=5)       # total questions
    percentage         = Column(Float, default=0.0)       # score/total * 100
    questions_json     = Column(JSON, nullable=True)      # full questions list
    user_answers_json  = Column(JSON, nullable=True)      # student's answers
    feedback           = Column(Text, nullable=True)      # LLM feedback
    timestamp          = Column(DateTime, default=datetime.utcnow)

    student = relationship("User", back_populates="quiz_attempts", foreign_keys=[student_id])
    session = relationship("StudentSession", back_populates="quiz_attempts")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id           = Column(Integer, primary_key=True, index=True)
    student_id   = Column(Integer, ForeignKey("users.id"))
    session_id   = Column(Integer, ForeignKey("student_sessions.id"), nullable=True)
    role         = Column(String)        # "user" or "assistant"
    content      = Column(Text)
    timestamp    = Column(DateTime, default=datetime.utcnow)

    student = relationship("User", back_populates="chat_messages", foreign_keys=[student_id])
    session = relationship("StudentSession", back_populates="chat_messages")


class ExamMark(Base):
    __tablename__ = "exam_marks"

    id               = Column(Integer, primary_key=True, index=True)
    student_id       = Column(Integer, ForeignKey("users.id"))
    professor_id     = Column(Integer, ForeignKey("users.id"))
    exam_name        = Column(String)
    subject          = Column(String)
    marks_obtained   = Column(Float)
    total_marks      = Column(Float)
    percentage       = Column(Float)
    grade            = Column(String)
    conducted_on     = Column(DateTime)
    uploaded_at      = Column(DateTime, default=datetime.utcnow)

    student   = relationship("User", back_populates="exam_marks", foreign_keys=[student_id])
    professor = relationship("User", foreign_keys=[professor_id])