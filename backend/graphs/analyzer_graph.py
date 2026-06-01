from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
import json
import os
from dotenv import load_dotenv
from typing import Any
load_dotenv()


# ── Types ────────────────────────────────────────────────────

class AnalyzerState(TypedDict):
    chat_history: str        # formatted chat messages
    quiz_history: str        # formatted quiz attempts
    student_name: str
    current_level: str
    current_weak_topics: List[dict]
    analysis_result: dict


class AnalyzerOutput(BaseModel):
    weak_topics: List[Any] = Field(
        ..., description="Topics student is struggling with"
    )
    #understood_topics: List[str] = Field(
     #   ..., description="Topics student has mastered"
    #)
    learning_style: str = Field(
        ..., description="Detected style: visual, example, theory, mixed, reading_writing"
    )
    confidence_level: str = Field(
        ..., description="overall confidence: low, medium, high"
    )
    recommended_topic: str = Field(
        ..., description="Single best next topic to study"
    )
    level: str = Field(
        ..., description="Suggested level: beginner, intermediate, advanced"
    )
    summary: str = Field(
        ..., description="2-3 sentence human readable summary of this session"
    )


# ── Node ─────────────────────────────────────────────────────

def analyze_session(state: AnalyzerState):
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3    # low temp for consistent structured output
    )
    structured_model = model.with_structured_output(AnalyzerOutput)

    prompt = PromptTemplate(
        template="""
You are an expert learning analyst. Analyze this student's study session.

STUDENT: {student_name}
CURRENT LEVEL: {current_level}
CURRENT WEAK TOPICS: {current_weak_topics}

CHAT HISTORY THIS SESSION:
{chat_history}

QUIZ RESULTS THIS SESSION:
{quiz_history}

Your job:
1. Identify topics the student is STRUGGLING with based on:
   - Quiz questions they got wrong
   - Topics they asked for repeated help on in chat
   - Topics where they expressed confusion

2. Identify topics they UNDERSTOOD based on:
   - Quiz questions answered correctly
   - Topics they explained back correctly in chat
   - Topics with no confusion signals

3. Detect their LEARNING STYLE from how they responded:
   - Did they ask for examples? → example
   - Did they ask for diagrams/structure? → visual
   - Did they prefer theory first? → theory
   - Mix of everything? → mixed

4. Assess CONFIDENCE LEVEL:
   - low: score < 50%, lots of confusion in chat
   - medium: score 50-75%, some confusion
   - high: score > 75%, minimal confusion

5. Recommend ONE specific next topic to study

6. Suggest appropriate difficulty level going forward

Be precise. Base everything on actual evidence from the chat and quiz data.
""",
        input_variables=[
            "student_name", "current_level", "current_weak_topics",
            "chat_history", "quiz_history"
        ]
    )

    chain = prompt | structured_model
    result = chain.invoke({
        "student_name": state["student_name"],
        "current_level": state["current_level"],
        "current_weak_topics": str(state["current_weak_topics"]),
        "chat_history": state["chat_history"],
        "quiz_history": state["quiz_history"]
    })

    return {
        "analysis_result": {
            "weak_topics": result.weak_topics,
            #"understood_topics": result.understood_topics,
            "learning_style": result.learning_style,
            "confidence_level": result.confidence_level,
            "recommended_topic": result.recommended_topic,
            "level": result.level,
            "summary": result.summary
        }
    }


# ── Build graph ──────────────────────────────────────────────

def build_analyzer_graph():
    graph = StateGraph(AnalyzerState)
    graph.add_node("analyze_session", analyze_session)
    graph.add_edge(START, "analyze_session")
    graph.add_edge("analyze_session", END)
    return graph.compile()


analyzer_flow = build_analyzer_graph()


# ── Public API ───────────────────────────────────────────────

def run_analyzer(
    student_name: str,
    current_level: str,
    current_weak_topics: list,
    chat_messages: list,
    quiz_attempts: list
) -> dict:
    # format chat and quiz history same as before
    chat_lines = []
    for msg in chat_messages:
        role = "Student" if msg.role == "user" else "Tutor"
        chat_lines.append(f"{role}: {msg.content}")
    chat_history = "\n".join(chat_lines) if chat_lines else "No messages."

    quiz_lines = []
    for attempt in quiz_attempts:
        quiz_lines.append(
            f"- {attempt.topic} | {attempt.score}/{attempt.total} "
            f"({attempt.percentage:.0f}%) | {attempt.difficulty}"
        )
        if attempt.questions_json and attempt.user_answers_json:
            for q, ans in zip(attempt.questions_json, attempt.user_answers_json):
                correct = q.get("correct_answer", "")
                if ans.strip().lower() != correct.strip().lower():
                    quiz_lines.append(
                        f"  Wrong: {q.get('question','')[:60]}..."
                    )
    quiz_history = "\n".join(quiz_lines) if quiz_lines else "No quizzes."

    state = AnalyzerState(
        chat_history=chat_history,
        quiz_history=quiz_history,
        student_name=student_name,
        current_level=current_level,
        current_weak_topics=current_weak_topics,
        analysis_result={}
    )

    result = analyzer_flow.invoke(state)
    return result["analysis_result"]