from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_cohere import CohereEmbeddings

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TEST_CASES = 10

ZERO_SCORES = {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0,
}


@lru_cache(maxsize=1)
def _get_embedding_model() -> CohereEmbeddings:
    return CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model="embed-english-v3.0"
        )


@lru_cache(maxsize=1)
def _get_groq_client() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_contexts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (_clean_text(item) for item in value) if text]


def _clamp_score(score: float) -> float:
    if not np.isfinite(score):
        return 0.0
    return float(max(0.0, min(1.0, score)))


def _cosine_text_similarity(left: str, right: str) -> float:
    left = _clean_text(left)
    right = _clean_text(right)

    if not left or not right:
        raise ValueError("Cosine similarity requires two non-empty strings.")

    embeddings = _get_embedding_model().embed_documents([left, right])
    similarity = cosine_similarity(
        np.asarray([embeddings[0]]),
        np.asarray([embeddings[1]]),
    )[0][0]

    return _clamp_score(float(similarity))


def _parse_faithfulness_score(response: Any) -> float:
    content = _clean_text(getattr(response, "content", response))
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", content)

    if not match:
        raise ValueError(f"Could not parse faithfulness score from response: {content!r}")

    return _clamp_score(float(match.group(0)))


def _calculate_faithfulness(answer: str, contexts: list[str]) -> float:
    context = "\n\n".join(contexts)

    if not answer or not context:
        raise ValueError("Faithfulness requires a non-empty answer and context.")

    prompt = f"""
Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate only whether the answer is supported by the context.

Return a single number between 0.0 and 1.0.

0.0 = unsupported or contradicts context
0.5 = partially supported
1.0 = fully supported

Output ONLY the number.
""".strip()

    response = _get_groq_client().invoke(prompt)
    return _parse_faithfulness_score(response)


def _calculate_answer_relevancy(question: str, answer: str) -> float:
    return _cosine_text_similarity(question, answer)


def _calculate_context_precision(question: str, contexts: list[str]) -> float:
    if not contexts:
        raise ValueError("Context precision requires at least one context chunk.")

    chunk_scores = [_cosine_text_similarity(question, chunk) for chunk in contexts]
    return _clamp_score(float(np.mean(chunk_scores)))


def _calculate_context_recall(answer: str, contexts: list[str]) -> float:
    combined_context = "\n\n".join(contexts)
    return _cosine_text_similarity(answer, combined_context)


def _evaluate_case(test_case: dict[str, Any], index: int) -> dict[str, float] | None:
    question = _clean_text(test_case.get("question"))
    answer = _clean_text(test_case.get("answer"))
    contexts = _clean_contexts(test_case.get("contexts"))

    if not question or not answer or not contexts:
        logger.warning(
            "Skipping RAG evaluation case %s: question, answer, and contexts are required.",
            index,
        )
        return None

    scores: dict[str, float] = {}

    try:
        scores["faithfulness"] = _calculate_faithfulness(answer, contexts)
    except Exception:
        logger.exception("Faithfulness calculation failed for RAG evaluation case %s.", index)
        return None

    try:
        scores["answer_relevancy"] = _calculate_answer_relevancy(question, answer)
    except Exception:
        logger.exception("Answer relevancy calculation failed for RAG evaluation case %s.", index)
        return None

    try:
        scores["context_precision"] = _calculate_context_precision(question, contexts)
    except Exception:
        logger.exception("Context precision calculation failed for RAG evaluation case %s.", index)
        return None

    try:
        scores["context_recall"] = _calculate_context_recall(answer, contexts)
    except Exception:
        logger.exception("Context recall calculation failed for RAG evaluation case %s.", index)
        return None

    return scores


def run_rag_evaluation(test_cases: list) -> dict:
    if not isinstance(test_cases, list) or not test_cases:
        return ZERO_SCORES.copy()

    successful_scores: list[dict[str, float]] = []

    for index, test_case in enumerate(test_cases[:MAX_TEST_CASES]):
        if not isinstance(test_case, dict):
            logger.warning("Skipping RAG evaluation case %s: case must be a dictionary.", index)
            continue

        try:
            case_scores = _evaluate_case(test_case, index)
            if case_scores is not None:
                successful_scores.append(case_scores)
        except Exception:
            logger.exception("Unexpected RAG evaluation failure for case %s.", index)

    if not successful_scores:
        return ZERO_SCORES.copy()

    final_scores = {}
    for metric_name in ZERO_SCORES:
        try:
            metric_scores = [scores[metric_name] for scores in successful_scores]
            final_scores[metric_name] = round(_clamp_score(float(np.mean(metric_scores))), 3)
        except Exception:
            logger.exception("Failed to aggregate RAG evaluation metric %s.", metric_name)
            final_scores[metric_name] = 0.0

    return final_scores
