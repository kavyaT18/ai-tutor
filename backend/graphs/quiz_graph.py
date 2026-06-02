from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict, Literal
import os
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_helper import RAGHelper

load_dotenv()

from rag_helper import get_rag
memory = MemorySaver()


# ── Types ────────────────────────────────────────────────────

class QuizQuestion(TypedDict):
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    source: Optional[str]

class QuizState(TypedDict):
    topic: str
    message: str
    knowledge_level: Literal["beginner", "intermediate", "advanced", "expert"]
    learning_style: Literal["visual", "auditory", "kinesthetic", "reading_writing", "mixed"]
    difficulty_level: Literal["easy", "medium", "hard"]
    num_questions: int
    retrieved_context: str
    questions: List[QuizQuestion]
    user_answers: List[str]
    score: int
    feedback: str
    curriculum_context: str

class QuizOutput(BaseModel):
    questions: List[QuizQuestion] = Field(
        ...,
        description="List of quiz questions with options, correct answer, explanation and source."
    )


# ── Nodes ────────────────────────────────────────────────────

def retrieve_context(state: QuizState):
    # More specific query — include subject context
    query = f"educational content about {state['topic']} for quiz questions"
    chunks = rag.query(query, k=6)

    # Filter chunks — only keep ones that mention the topic
    topic_words = state["topic"].lower().split()
    filtered = []
    for chunk in chunks:
        content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        content_lower = content.lower()
        # Keep chunk only if at least one topic word appears in it
        if any(word in content_lower for word in topic_words if len(word) > 3):
            filtered.append(content)

    # Fallback to all chunks if filter removes everything
    if not filtered:
        filtered = [
            chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            for chunk in chunks
        ]

    retrieved_context = "\n\n".join(filtered[:4])
    return {"retrieved_context": retrieved_context}


def create_quiz(state: QuizState):
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    structured_model = model.with_structured_output(QuizOutput)

    prompt = PromptTemplate(
        template="""
You are an expert educational quiz generator.
Generate a high-quality quiz based primarily on the retrieved course material.
IMPORTANT: Only generate questions strictly about "{topic}".
If the retrieved material does not contain enough information 
about {topic}, generate questions from your own knowledge about {topic}.
Do NOT use material about unrelated topics even if it appears in the context.

TOPIC: {topic}
COURSE MATERIAL: {retrieved_context}
DIFFICULTY LEVEL: {difficulty_level}
STUDENT KNOWLEDGE LEVEL: {knowledge_level}
NUMBER OF QUESTIONS: {num_questions}
ADDITIONAL USER INSTRUCTIONS: {message}
CURRICULUM CONTEXT:
{curriculum_context}

Only generate questions on topics that exist in this curriculum.
The topic "{topic}" belongs to this curriculum.
Stay strictly within this subject area.

Instructions:
1. Use retrieved course material as primary source.
if the topic is strictly not related to the retrieved course materials,dont give questions based on the retrieved course materials and instead rely on your domain knowledge to generate the quiz questions.
2. If material is incomplete, extend with domain knowledge.
3. Generate exactly {num_questions} questions.
4. Adapt to {knowledge_level} learner at {difficulty_level} difficulty.
5. Cover multiple concepts, not the same idea repeatedly.
6. Include conceptual and application-based questions.
7. Questions progress from easier to harder.
8. Each question has exactly 4 answer options.
9. Only one option is correct and must exactly match correct_answer.
10. Include a clear explanation for every answer.
11. Avoid duplicate or overly similar questions.
12. Create realistic distractor options.

Return strictly according to the schema.
""",
        input_variables=[
            "topic", "retrieved_context", "difficulty_level",
            "knowledge_level", "num_questions", "message","curriculum_context"
        ]
    )

    chain = prompt | structured_model
    result = chain.invoke({
        "topic": state["topic"],
        "knowledge_level": state["knowledge_level"],
        "difficulty_level": state["difficulty_level"],
        "num_questions": state["num_questions"],
        "message": state["message"],
        "retrieved_context": state["retrieved_context"],
        "curriculum_context": state["curriculum_context"]
    })
    return {"questions": result.questions}
def load_curriculum(state: QuizState):
    import json
    import os

    CURRICULUM_FILE = "curriculum.json"

    if not os.path.exists(CURRICULUM_FILE):
        return {"curriculum_context": ""}

    try:
        with open(CURRICULUM_FILE, "r") as f:
            curriculum = json.load(f)

        if not curriculum:
            return {"curriculum_context": ""}

        lines = []
        for e in curriculum:
            lines.append(
                f"Subject: {e['subject']} | Topics: {', '.join(e.get('topics', []))}"
            )

        return {"curriculum_context": "\n".join(lines)}

    except Exception as e:
        print("Curriculum load error:", e)
        return {"curriculum_context": ""}

def ask_questions(state: QuizState):
    """Pause graph here — frontend sends answers via /quiz/answer"""
    quiz = [
        {
            "number": idx,
            "question": q["question"],
            "options": q["options"]
        }
        for idx, q in enumerate(state["questions"], start=1)
    ]
    answers = interrupt({"quiz": quiz})
    return {"user_answers": answers}


def evaluate_answers(state: QuizState):
    score = 0
    for question, answer in zip(state["questions"], state["user_answers"]):
        if answer.strip().lower() == question["correct_answer"].strip().lower():
            score += 1
    return {"score": score}


def generate_feedback(state: QuizState):
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template="""
You are an expert tutor and learning coach.
Analyze the student's quiz performance and provide personalized feedback.

TOPIC: {topic}
STUDENT KNOWLEDGE LEVEL: {knowledge_level}
SCORE: {score}/{total_questions}
QUIZ DETAILS: {questions}
STUDENT ANSWERS: {user_answers}

Your tasks:
1. Summarize overall performance.
2. Identify concepts student understands well.
3. Identify concepts where student struggled.
4. Explain common mistakes.
5. Recommend specific topics for further study.
6. Provide 3 actionable improvement suggestions.
7. Estimate whether student is ready for harder difficulty.
8. Maintain an encouraging and constructive tone.

Format:
Overall Assessment: ...
Strengths: ...
Areas for Improvement: ...
Recommended Topics: ...
Study Plan: 1. ... 2. ... 3. ...
Difficulty Recommendation: ...
""",
        input_variables=[
            "topic", "knowledge_level", "score",
            "total_questions", "questions", "user_answers"
        ]
    )

    chain = prompt | model
    result = chain.invoke({
        "topic": state["topic"],
        "knowledge_level": state["knowledge_level"],
        "score": state["score"],
        "total_questions": len(state["questions"]),
        "questions": state["questions"],
        "user_answers": state["user_answers"]
    })
    return {"feedback": result.content}


# ── Build graph ──────────────────────────────────────────────

def build_quiz_graph():
    graph = StateGraph(QuizState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("create_quiz", create_quiz)
    graph.add_node("ask_questions", ask_questions)
    graph.add_node("evaluate_answers", evaluate_answers)
    graph.add_node("generate_feedback", generate_feedback)
    graph.add_node("load_curriculum", load_curriculum)
    graph.add_edge(START, "load_curriculum")
    graph.add_edge("load_curriculum", "retrieve_context")  # was START -> retrieve_context

   
    graph.add_edge("retrieve_context", "create_quiz")
    graph.add_edge("create_quiz", "ask_questions")
    graph.add_edge("ask_questions", "evaluate_answers")
    graph.add_edge("evaluate_answers", "generate_feedback")
    graph.add_edge("generate_feedback", END)

    return graph.compile(checkpointer=memory)


quiz_flow = build_quiz_graph()


# ── Public API used by agent_handler ────────────────────────

def start_quiz(
    thread_id: str,
    topic: str,
    knowledge_level: str = "beginner",
    learning_style: str = "mixed",
    difficulty_level: str = "medium",
    num_questions: int = 5,
    message: str = ""
) -> dict:
    """
    Start quiz. Returns questions for frontend.
    Graph pauses at ask_questions interrupt.
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = QuizState(
        topic=topic,
        message=message,
        knowledge_level=knowledge_level,
        learning_style=learning_style,
        difficulty_level=difficulty_level,
        num_questions=num_questions,
        retrieved_context="",
        questions=[],
        user_answers=[],
        score=0,
        feedback=""
    )

    result = quiz_flow.invoke(initial_state, config=config)

    # Graph is paused at interrupt — extract questions
    interrupt_data = result.get("__interrupt__", [{}])[0]
    questions = interrupt_data.value.get("quiz", []) if hasattr(interrupt_data, 'value') else []

    return {
        "thread_id": thread_id,
        "questions": questions,
        "status": "awaiting_answers"
    }


def submit_answers(thread_id: str, answers: List[str]) -> dict:
    """
    Resume graph with student answers.
    Returns score, feedback, full questions with correct answers.
    """
    config = {"configurable": {"thread_id": thread_id}}

    final_result = quiz_flow.invoke(
        Command(resume=answers),
        config=config
    )

    return {
        "score": final_result.get("score", 0),
        "total": len(final_result.get("questions", [])),
        "percentage": round(
            final_result.get("score", 0) /
            max(len(final_result.get("questions", [])), 1) * 100, 1
        ),
        "feedback": final_result.get("feedback", ""),
        "questions": final_result.get("questions", []),
        "user_answers": final_result.get("user_answers", [])
    }