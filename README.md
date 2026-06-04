# AdaptLearn — Agentic AI Tutor Platform



> An end-to-end agentic AI tutoring platform that turns static course PDFs into a living, adaptive learning experience — built with RAG, LangGraph multi-agent orchestration, and real-time student analytics.

---

## What is AdaptLearn?

Most AI chatbots answer questions generically. AdaptLearn is different.

When a student asks "explain recursion," AdaptLearn already knows they scored 38% on their last algorithms quiz, that they prefer example-first explanations, and that their course material defines recursion through a specific textbook chapter. The response is pulled from the actual uploaded course PDFs, calibrated to the student's knowledge level, and informed by their recent performance history.

Professors upload course PDFs once. The system automatically chunks, embeds, and stores them in a vector database. Every student interaction — tutoring conversation or quiz — is grounded in actual course content rather than the LLM's general knowledge.

---

## Live Demo

Backend deployed on Render. Frontend deployed on Netlify.

> Register as a student or professor to explore the platform.

---

## Key Features

### For Students
- Adaptive AI tutor chat with sidebar session history, full conversation memory across sessions
- RAG-powered answers grounded in professor-uploaded course PDFs
- Quiz generation on any topic with MCQs, explanations, and detailed feedback
- Weak topic tracking mapped against professor-defined curriculum using LLM matching
- Progress dashboard with score history charts, topic performance bars, risk score meter
- Session analyzer that detects learning style, weak topics, and confidence level at session end
- Exam marks view showing results uploaded by professor

### For Professors
- Course PDF upload with automatic chunking, embedding, and storage in ChromaDB
- Curriculum management — define subjects and topics that guide weak topic detection
- Student analytics dashboard with individual drilldown, score history, and risk scores
- Exam marks upload via single entry or batch CSV with auto grade computation
- At-risk student alerts for students crossing risk threshold
- RAG evaluation dashboard showing faithfulness, answer relevancy, context precision, context recall

---



## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| AI Orchestration | LangGraph |
| LLM | Groq / Llama 3.3 70B |
| Embeddings | Cohere embed-english-v3.0 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Database | PostgreSQL via Neon |
| Conversation Memory | LangGraph SqliteSaver |
| Risk Model | XGBoost |
| RAG Evaluation | RAGAS |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Render (backend) + Netlify (frontend) |

---





## RAG Pipeline

**Ingestion**
1. PDF loaded page by page via `PyPDFLoader`
2. Text split into 500-token chunks with 50-token overlap
3. Each chunk sent to Cohere `embed-english-v3.0` API returning 1024-dimensional vectors
4. Vectors stored in ChromaDB with source metadata including filename and page number

**Retrieval**
1. Student query embedded using same Cohere model
2. ChromaDB cosine similarity search returns top-k chunks
3. Chunks filtered to keep only those relevant to the query topic
4. Retrieved chunks injected into LLM system prompt alongside student profile and quiz history
5. Groq generates a response grounded in course material with source citations

---

## RAG Evaluation Metrics

| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.95 | Answer claims supported by retrieved context, not hallucinated |
| Answer Relevancy | 0.673 | Answer is relevant to the question asked |
| Context Precision | 0.463 | Retrieved chunks are genuinely useful for answering |
| Context Recall | 0.666 | Retrieval captured all information needed to answer correctly |

```
Knowledge Base     524+ vectorized chunks
Faithfulness       0.95  ████████████████████░
Answer Relevancy   0.673 █████████████░░░░░░░░
Context Precision  0.463 █████████░░░░░░░░░░░░
Context Recall     0.666 █████████████░░░░░░░░
Avg Latency        ~3.0 seconds per response
```

---

## LangGraph Agents

**Chat Graph — 7 nodes**
```
load_student_profile → load_quiz_history → load_curriculum
→ determine_intent → retrieve_rag → web_search
→ generate_response → save_to_db
```

**Quiz Graph — 6 nodes**
```
load_curriculum → determine_intent → retrieve_context
→ create_quiz → ask_questions [INTERRUPT] → evaluate_answers
→ generate_feedback
```

**Analyzer Graph — runs at session end**
```
analyze_session → returns weak_topics, learning_style,
confidence_level, recommended_topic, level, summary
```

---

## Project Structure

```
ai-tutor/
│
├── backend/
│   ├── graphs/
│   │   ├── chat_graph.py         # LangGraph tutor with 7 nodes
│   │   ├── quiz_graph.py         # LangGraph quiz with interrupt pattern
│   │   └── analyzer_graph.py    # Session end analyzer graph
│   ├── main.py                   # FastAPI app and all routes
│   ├── database.py               # SQLAlchemy engine and session
│   ├── models.py                 # All DB table definitions
│   ├── auth.py                   # JWT token handling
│   ├── agent_handler.py          # Connects routes to graphs
│   ├── rag_helper.py             # RAGHelper with lazy initialization
│   ├── rag_evaluator.py          # RAGAS evaluation runner
│   ├── rag_singleton.py          # lru_cache singleton for RAGHelper (removed torch dependency)
│   ├── topic_matcher.py          # LLM-based curriculum topic matching
│   ├── risk_engine.py            # Feature extraction and scoring
│   ├── risk_model.py             # XGBoost model wrapper
│   ├── risk_tasks.py             # Triggered risk rescoring
│   ├── schemas.py                # Pydantic request schemas
│   ├── study_agents.py           # Legacy agent reference
│   ├── prompts.yaml              # Centralized prompt templates
│   └── requirements.txt
│
└── frontend/
    ├── config.js                 # Central API URL config
    ├── index.html                # Login and register
    ├── student.html              # Student dashboard
    ├── chat.html                 # Chat with sidebar session history
    ├── chat-history.html         # Past sessions list
    ├── quiz.html                 # Quiz with topic autocomplete
    ├── quiz-history.html         # Past quizzes with feedback
    ├── progress.html             # Score charts and weak topics
    ├── profile.html              # Level and learning style settings
    ├── professor.html            # Professor dashboard with tabs
    ├── exams.html                # Exam marks upload and analytics
    ├── upload.html               # PDF upload
    └── style.css                 # Shared styles
```

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database (local or Neon)
- Groq API key — console.groq.com
- Cohere API key — dashboard.cohere.com

### Setup

```bash
git clone https://github.com/kavyaT18/ai-tutor.git
cd ai-tutor/backend

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables

Create `backend/.env`:

```env
DATABASE_URL=postgresql://user:password@host/dbname
SECRET_KEY=your_secret_key_here
GROQ_API_KEY=gsk_your_groq_key
COHERE_API_KEY=your_cohere_key
TOKENIZERS_PARALLELISM=false
```

---

## Running Locally

```bash
cd backend
uvicorn main:app --reload
```

Open `frontend/index.html` in your browser.

To point the frontend at localhost, update `frontend/config.js`:

```javascript
const API = "http://localhost:8000"
```

---

## Deployment Notes

- Backend deployed on Render free tier
- Frontend deployed on Netlify
- HuggingFace sentence-transformers replaced with Cohere API embeddings to stay within Render's 512MB RAM limit
- RAGHelper uses lazy initialization — embeddings load only on first actual RAG request, not at startup
- ChromaDB uses local filesystem storage — PDFs must be re-uploaded after Render redeploys (ephemeral filesystem)
- LangGraph checkpoints stored in SqliteSaver for conversation persistence

---

## Future Improvements

- Adaptive quiz difficulty based on recent performance trend
- Chroma Cloud for persistent vector storage across deploys
- Multi-professor scoping so each professor's PDFs are visible only to their students
- Email alerts when a student's risk score crosses threshold
- Retrain XGBoost risk model once real student data accumulates
- Mobile-responsive frontend

---

## License

MIT License. See `LICENSE` for details.

---

<p align="center">Built with FastAPI, LangGraph, Groq, ChromaDB, and a lot of debugging.</p>
