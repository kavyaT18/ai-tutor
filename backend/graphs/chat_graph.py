import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_helper import RAGHelper
from dotenv import load_dotenv
from typing import Any
load_dotenv()

from rag_helper import get_rag
search_tool = DuckDuckGoSearchRun()

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints.db")

EDUCATIONAL_KEYWORDS = [
    "explain", "teach", "summarize", "what is", "how does",
    "define", "concept", "example", "notes", "topic",
    "difference between", "why is", "how to"
]

WEB_KEYWORDS = [
    "latest", "recent", "current", "today", "new",
    "trending", "2024", "2025", "news"
]


class TutorState(TypedDict):
    messages: Annotated[List, add_messages]
    student_id: int
    thread_id: str
    message: str
    knowledge_level: str
    learning_style: str
    curriculum_context: str
    

    weak_topics: List[Any]
   # understood_topics: List[str]
    quiz_history: str
    retrieved_context: str
    web_context: str
    sources: List[str]
    response: str


def load_student_profile(state: TutorState):
    from database import SessionLocal
    import models
    db = SessionLocal()
    try:
        profile = db.query(models.StudentProfile).filter_by(
            user_id=state["student_id"]
        ).first()
        if profile:
            return {
                "knowledge_level": profile.level or "beginner",
                "learning_style": profile.learning_style or "mixed",
                "weak_topics": profile.weak_topics or [],
              #  "understood_topics": profile.understood_topics or []
            }
    finally:
        db.close()
    return {
        "knowledge_level": "beginner",
        "learning_style": "mixed",
        "weak_topics": [],
       # "understood_topics": []
    }

CURRICULUM_FILE = "curriculum.json"

def load_curriculum(state):

    curriculum = []

    if os.path.exists(CURRICULUM_FILE):
        try:
            with open(CURRICULUM_FILE, "r") as f:
                curriculum = json.load(f)
        except Exception:
            curriculum = []

    return {
        "curriculum": curriculum
    }

def load_quiz_history(state: TutorState):
    from database import SessionLocal
    import models
    db = SessionLocal()
    try:
        attempts = (
            db.query(models.QuizAttempt)
            .filter_by(student_id=state["student_id"])
            .order_by(models.QuizAttempt.timestamp.desc())
            .limit(5)
            .all()
        )
        if not attempts:
            return {"quiz_history": "No quiz attempts yet."}

        lines = ["Recent quiz performance:"]
        for a in reversed(attempts):
            lines.append(
                f"- {a.topic} | {a.difficulty} | "
                f"{a.score}/{a.total} = {a.percentage:.0f}%"
            )
            if a.questions_json and a.user_answers_json:
                for q, ans in zip(a.questions_json, a.user_answers_json):
                    correct = q.get("correct_answer", "")
                    if ans.strip().lower() != correct.strip().lower():
                        lines.append(
                            f"  Wrong: {q.get('question','')[:60]}..."
                        )
        return {"quiz_history": "\n".join(lines)}
    finally:
        db.close()


def determine_intent(state: TutorState):
    msg = state["message"].lower()
    needs_rag = any(kw in msg for kw in EDUCATIONAL_KEYWORDS)
    needs_web = any(kw in msg for kw in WEB_KEYWORDS)
    return {
        "retrieved_context": "RAG_NEEDED" if needs_rag else "",
        "web_context": "WEB_NEEDED" if needs_web else ""
    }


def retrieve_rag(state: TutorState):
    if state.get("retrieved_context") != "RAG_NEEDED":
        return {}
    chunks = get_rag().query(state["message"], k=4)
    sources = []
    context_parts = []
    for chunk in chunks:
        if hasattr(chunk, "page_content"):
            context_parts.append(chunk.page_content)
            meta = getattr(chunk, "metadata", {})
            src = meta.get("source", "Course material")
            page = meta.get("page", "")
            sources.append(f"{src} Page {page}" if page else src)
        else:
            context_parts.append(str(chunk))
            sources.append("Course material")

    return {
        "retrieved_context": "\n\n".join(context_parts),
        "sources": sources
    }


def web_search(state: TutorState):
    if state.get("web_context") != "WEB_NEEDED":
        return {}
    try:
        result = search_tool.run(state["message"])
        return {
            "web_context": result,
            "sources": (state.get("sources") or []) + ["Web Search Result"]
        }
    except Exception:
        return {"web_context": ""}


def generate_response(state: TutorState):
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )

    weak_topics = state.get("weak_topics") or []

    weak = ", ".join(
    w["topic"] if isinstance(w, dict) else str(w)
    for w in weak_topics
) or "none"
    understood = ", ".join(state.get("understood_topics") or []) or "none"

    style_map = {
        "visual": "Use structured layouts and ASCII diagrams where helpful.",
        "example": "Always show a concrete example before explaining theory.",
        "theory": "Explain theory first, then show examples.",
        "reading_writing": "Use detailed written explanations with clear headings.",
        "auditory": "Explain conversationally as if speaking aloud.",
        "kinesthetic": "Focus on hands-on step-by-step walkthroughs.",
        "mixed": "Balance theory and examples naturally."
    }
    style_note = style_map.get(
        state.get("learning_style", "mixed"),
        "Balance theory and examples."
    )

    level = state.get("knowledge_level", "beginner")
    level_note = {
        "beginner": "Use simple words and real-life analogies. Avoid jargon.",
        "intermediate": "Use some technical terms with code examples.",
        "advanced": "Go deep. Cover edge cases and best practices.",
        "expert": "Assume strong prior knowledge. Be concise and precise."
    }.get(level, "Adapt to the student's level.")

    rag_ctx = state.get("retrieved_context", "")
    web_ctx = state.get("web_context", "")
    quiz_ctx = state.get("quiz_history", "")
    sources = state.get("sources") or []
    curriculum_ctx = state.get("curriculum_context", "")

    system = f"""You are a personalized AI tutor.

Student profile:
- Level: {level}
- Learning style: {state.get("learning_style", "mixed")}
- Weak topics: {weak}
- Mastered topics: {understood}
{f"Curriculum:{chr(10)}{curriculum_ctx}{chr(10)}" if curriculum_ctx else ""}

Instructions:
- {level_note}
- {style_note}
- If the student struggled with related topics recently, slow down and be extra clear.
-Only teach topics that exist in the curriculum above {curriculum_ctx}.
- If student asks about something outside curriculum, politely redirect.

{f"Recent quiz performance:{chr(10)}{quiz_ctx}{chr(10)}" if quiz_ctx else ""}
{f"Course material:{chr(10)}{rag_ctx}{chr(10)}" if rag_ctx and rag_ctx != "RAG_NEEDED" else ""}
{f"Web search results:{chr(10)}{web_ctx}{chr(10)}" if web_ctx and web_ctx != "WEB_NEEDED" else ""}
{f"At the end of your response, list sources:{chr(10)}" + chr(10).join(f'- {s}' for s in sources) if sources else ""}"""

    messages_to_send = [SystemMessage(content=system)] + state["messages"]

    response = model.invoke(messages_to_send)
    return {
        "response": response.content,
        "messages": [response]
    }


def save_to_db(state: TutorState):
    from database import SessionLocal
    import models
    db = SessionLocal()
    try:
        # Extract session_id from thread_id format: chat_{student_id}_{session_id}
        parts = state["thread_id"].split("_")
        session_id = int(parts[-1]) if len(parts) >= 3 else None

        db.add(models.ChatMessage(
            student_id=state["student_id"],
            session_id=session_id,
            role="user",
            content=state["message"]
        ))
        db.add(models.ChatMessage(
            student_id=state["student_id"],
            session_id=session_id,
            role="assistant",
            content=state["response"]
        ))
        db.commit()
    finally:
        db.close()
    return {}


# Build graph
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# global connection
conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

def build_chat_graph():
    graph = StateGraph(TutorState)

    graph.add_node("load_student_profile", load_student_profile)
    graph.add_node("load_quiz_history", load_quiz_history)
    graph.add_node("determine_intent", determine_intent)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_to_db", save_to_db)
    graph.add_node("load_curriculum", load_curriculum)

    graph.add_edge(START, "load_student_profile")
    graph.add_edge("load_student_profile", "load_quiz_history")
    graph.add_edge("load_quiz_history", "load_curriculum")      # add this
    graph.add_edge("load_curriculum", "determine_intent")  
   
    graph.add_edge("determine_intent", "retrieve_rag")
    graph.add_edge("retrieve_rag", "web_search")
    graph.add_edge("web_search", "generate_response")
    graph.add_edge("generate_response", "save_to_db")
    graph.add_edge("save_to_db", END)

    return graph.compile(checkpointer=checkpointer)

chat_flow = build_chat_graph()


def send_message(
    thread_id: str,
    message: str,
    student_id: int
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    # Only pass what's new — checkpointer handles history automatically
    state = {
        "messages": [HumanMessage(content=message)],
        "student_id": student_id,
        "thread_id": thread_id,
        "message": message,
        # These get loaded fresh each time by the graph nodes
        "knowledge_level": "",
        "learning_style": "",
        "weak_topics": [],
       # "understood_topics": [],
        "quiz_history": "",
        "retrieved_context": "",
        "web_context": "",
        "sources": [],
        "response": "",
        "curriculum_context": ""
    }

    result = chat_flow.invoke(state, config=config)
    return {
        "response": result.get("response", ""),
        "sources": result.get("sources") or []
    }