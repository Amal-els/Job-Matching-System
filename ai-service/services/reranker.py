import json
import logging
import os

from services.llm_client import chat_json, get_llm_client, is_enabled


def _fallback_rerank_jobs(jobs, limit=10):
    reranked = []

    for job in jobs:
        semantic_score = float(job.get("semantic_similarity", 0.0))
        keyword_score = float(job.get("keyword_score", 0.0))
        recency_score = float(job.get("recency_score", 0.0))
        proximity_score = float(job.get("proximity_score", 0.0))
        final_score = (
            (0.45 * semantic_score)
            + (0.30 * keyword_score)
            + (0.15 * recency_score)
            + (0.10 * proximity_score)
        )

        reasons = []
        matched_skills = job.get("matched_skills", [])
        if matched_skills:
            reasons.append(f"Matched skills: {', '.join(matched_skills[:4])}")
        if job.get("seniority"):
            reasons.append(f"Detected seniority: {job['seniority']}")
        if job.get("industry"):
            reasons.append(f"Industry: {job['industry']}")
        if job.get("remote"):
            reasons.append("Remote-friendly role")

        reranked.append({
            **job,
            "final_score": round(final_score, 4),
            "rerank_mode": "weighted_fallback",
            "match_summary": reasons[:3],
        })

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked[:limit]


def _profile_snapshot(profile):
    return {
        "id": profile.get("id"),
        "bio": profile.get("bio"),
        "user_level": profile.get("user_level"),
        "profil_score": profile.get("profil_score"),
        "skills": profile.get("skills"),
        "experiences": profile.get("experiences"),
        "education": profile.get("education"),
        "languages": profile.get("languages"),
        "certifications": profile.get("certifications"),
        "target_position": profile.get("target_position"),
        "short_term_goals": profile.get("short_term_goals"),
        "long_term_goals": profile.get("long_term_goals"),
    }


def _job_snapshot(job):
    return {
        "id": job.get("id"),
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "remote": job.get("remote"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "contract_type": job.get("contract_type"),
        "description": job.get("description"),
        "skills_required": job.get("skills_required"),
        "source": job.get("source"),
        "posted_at": job.get("posted_at"),
        "url": job.get("url"),
        "seniority": job.get("seniority"),
        "industry": job.get("industry"),
        "matched_skills": job.get("matched_skills"),
        "keyword_score": job.get("keyword_score"),
        "semantic_similarity": job.get("semantic_similarity"),
        "recency_score": job.get("recency_score"),
        "proximity_score": job.get("proximity_score"),
    }


def _normalize_llm_ranking(payload):
    if isinstance(payload, dict):
        ranking = payload.get("ranking") or payload.get("jobs") or []
    else:
        ranking = payload

    normalized = []
    for item in ranking or []:
        if not isinstance(item, dict):
            continue

        job_id = item.get("job_id") or item.get("id")
        if not job_id:
            continue

        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0

        reasons = item.get("reasons") or item.get("match_summary") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]

        normalized.append({
            "job_id": str(job_id),
            "score": max(0.0, min(score, 1.0)),
            "reasons": [str(reason).strip() for reason in reasons if str(reason).strip()],
        })

    return normalized


def _llm_rerank_jobs(profile, jobs, limit=10):
    model_name = os.getenv("RERANKER_MODEL_NAME", "openai/gpt-oss-120b:free")
    prompt_payload = {
        "profile": _profile_snapshot(profile),
        "jobs": [_job_snapshot(job) for job in jobs],
        "top_k": limit,
    }
    prompt = f"""
Rank these jobs for the candidate profile and return the best matches.

Return only valid JSON in this exact format:
{{
  "ranking": [
    {{
      "job_id": "job identifier",
      "score": 0.0,
      "reasons": ["short reason 1", "short reason 2"]
    }}
  ]
}}

Rules:
- Return at most top_k jobs.
- Prefer strong fit on target_position, skills, experiences, education, certifications, languages, and stated goals.
- Use recency and proximity as secondary signals.
- `score` must be between 0 and 1.
- `reasons` should be short and concrete.
- Do not return markdown or commentary outside JSON.

Input:
{json.dumps(prompt_payload, ensure_ascii=True)}
"""

    response = chat_json(
        model=model_name,
        system_prompt="You are a careful recruiting reranker. Return strict JSON only.",
        user_prompt=prompt,
        temperature=0,
    )
    ranking = _normalize_llm_ranking(response)
    ranking_by_id = {item["job_id"]: item for item in ranking}

    reranked = []
    for job in jobs:
        llm_item = ranking_by_id.get(str(job.get("id")))
        if not llm_item:
            continue

        reranked.append({
            **job,
            "final_score": round(llm_item["score"], 4),
            "rerank_mode": "llm",
            "match_summary": llm_item["reasons"][:3],
        })

    reranked.sort(key=lambda item: item["final_score"], reverse=True)
    return reranked[:limit]


logger = logging.getLogger(__name__)


def rerank_jobs(profile, jobs, limit=10):
    should_use_llm = is_enabled("USE_LLM_RERANKER", "true")
    if should_use_llm and get_llm_client():
        try:
            llm_ranked = _llm_rerank_jobs(profile, jobs, limit=limit)
            if llm_ranked:
                return llm_ranked
            logger.warning("LLM reranker returned empty results, falling back.")
        except Exception as exc:
            logger.error("LLM reranker failed: %s", exc, exc_info=True)

    return _fallback_rerank_jobs(jobs, limit=limit)
