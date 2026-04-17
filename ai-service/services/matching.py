import re
import os
import json
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from time import perf_counter

from sqlalchemy import or_, and_
from models.job import Job
from ingestion.adzuna import fetch_adzuna_jobs
from ingestion.mapper import map_adzuna_job
from services.embedding import embed, embed_many
from services.job_text import build_job_text, build_profile_text, normalize_list
from services.job_parser import extract_job_metadata
from services.reranker import rerank_jobs
from services.similarity import cosine_top_k
from services.ann_index import query_ann_jobs
from services.cache import get_cache


FETCH_LIMIT = int(os.getenv("FETCH_LIMIT", "100"))
KEYWORD_TOP_K = int(os.getenv("KEYWORD_TOP_K", "30"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "15"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "10"))
REQUEST_TIMEOUT_MS = int(os.getenv("REQUEST_TIMEOUT_MS", "800"))
PROFILE_EMBED_TTL_SECONDS = int(os.getenv("PROFILE_EMBED_TTL_SECONDS", "3600"))
RESULT_CACHE_TTL_SECONDS = int(os.getenv("RESULT_CACHE_TTL_SECONDS", "600"))
JOB_TTL_DAYS = int(os.getenv("JOB_TTL_DAYS", "14"))


def fetch_jobs_hybrid(db, query, country="fr", limit=50, location=None):
    """
    Search DB first for fresh jobs, then enrich from Adzuna if needed.
    """
    if db is None:
        return fetch_adzuna_jobs(country=country, query=query, limit=limit, location=location)

    # 1. Search DB for fresh matches within TTL window (default: 14 days)
    threshold = datetime.now(timezone.utc) - timedelta(days=JOB_TTL_DAYS)
    
    # Simple semantic search using ILIKE on title for keywords
    # This is a basic fallback before we use vector search on the DB directly
    db_jobs_query = db.query(Job).filter(
        and_(
            or_(Job.title.ilike(f"%{query}%"), Job.description.ilike(f"%{query}%")),
            Job.created_at >= threshold
        )
    )
    
    db_jobs = db_jobs_query.limit(limit).all()
    results = [job_to_dict(job) for job in db_jobs]
    
    # 2. If not enough, fetch from Adzuna
    if len(results) < limit:
        remaining = limit - len(results)
        raw_api_jobs = fetch_adzuna_jobs(
            country=country, 
            query=query, 
            limit=remaining, 
            location=location
        )
        
        new_jobs = []
        for raw_job in raw_api_jobs:
            mapped = map_adzuna_job(raw_job) if raw_job.get("redirect_url") else raw_job
            # Avoid duplicate results in the final list if they were already in the DB results
            if not any(r["id"] == mapped["id"] for r in results):
                new_jobs.append(mapped)
        
        # 3. Cache new jobs to DB
        save_jobs_to_db(db, new_jobs)
        results.extend(new_jobs)

    return results[:limit]


def save_jobs_to_db(db, jobs):
    """
    Saves a list of mapped jobs to the database, ignoring duplicates.
    Computes and stores embeddings at ingest time so /match avoids per-job remote embeds.
    """
    if not db or not jobs:
        return

    to_insert = []
    for job_data in jobs:
        exists = db.query(Job).filter(Job.id == job_data["id"]).first()
        if exists:
            continue
        merged = ensure_job_metadata(job_data)
        job_data.update(merged)
        to_insert.append(job_data)

    if not to_insert:
        return

    texts = [build_job_text(job) for job in to_insert]
    embeddings = embed_many(texts)

    for job_data, job_embedding in zip(to_insert, embeddings):
        job_data["vector"] = job_embedding
        db.add(
            Job(
                id=job_data["id"],
                title=job_data["title"],
                company=job_data["company"],
                location=job_data["location"],
                remote=job_data["remote"],
                salary_min=job_data["salary_min"],
                salary_max=job_data["salary_max"],
                contract_type=job_data["contract_type"],
                description=job_data["description"],
                skills_required=job_data["skills_required"],
                source=job_data["source"],
                posted_at=job_data.get("posted_at"),
                url=job_data.get("url"),
                embedding=job_embedding,
            )
        )

    try:
        db.commit()
    except Exception:
        db.rollback()


def fetch_jobs_from_db(db, query, limit=50):
    if db is None:
        return []

    threshold = datetime.now(timezone.utc) - timedelta(days=JOB_TTL_DAYS)
    # Enforce TTL: delete stale jobs so new jobs can replace them.
    try:
        db.query(Job).filter(Job.created_at < threshold).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
    db_jobs_query = db.query(Job).filter(
        and_(
            or_(Job.title.ilike(f"%{query}%"), Job.description.ilike(f"%{query}%")),
            Job.created_at >= threshold
        )
    )
    db_jobs = db_jobs_query.limit(limit).all()
    return [job_to_dict(job) for job in db_jobs]


def fetch_jobs_from_adzuna(query, country="fr", limit=50, location=None):
    return fetch_adzuna_jobs(country=country, query=query, limit=limit, location=location)


async def fetch_jobs_hybrid_async(db, query, country="fr", limit=50, location=None):
    if db is None:
        external_started = perf_counter()
        api_jobs = await asyncio.to_thread(fetch_jobs_from_adzuna, query, country, limit, location)
        return api_jobs[:limit], {
            "db": 0,
            "external": len(api_jobs[:limit]),
            "db_ms": 0,
            "external_ms": int((perf_counter() - external_started) * 1000),
        }

    db_started = perf_counter()
    db_task = asyncio.to_thread(fetch_jobs_from_db, db, query, limit)
    external_started = perf_counter()
    external_task = asyncio.to_thread(fetch_jobs_from_adzuna, query, country, limit, location)
    db_jobs, api_jobs = await asyncio.gather(db_task, external_task, return_exceptions=True)
    db_elapsed = int((perf_counter() - db_started) * 1000)
    external_elapsed = int((perf_counter() - external_started) * 1000)

    if isinstance(db_jobs, Exception):
        db_jobs = []
    if isinstance(api_jobs, Exception):
        api_jobs = []

    mapped_api_jobs = []
    for raw_job in api_jobs:
        mapped = map_adzuna_job(raw_job) if raw_job.get("redirect_url") else raw_job
        mapped_api_jobs.append(mapped)

    results = list(db_jobs)
    for mapped in mapped_api_jobs:
        if not any(r["id"] == mapped["id"] for r in results):
            results.append(mapped)
        if len(results) >= limit:
            break

    save_jobs_to_db(db, mapped_api_jobs)
    return results[:limit], {
        "db": len(db_jobs),
        "external": len(mapped_api_jobs),
        "db_ms": db_elapsed,
        "external_ms": external_elapsed,
    }


def job_to_dict(job):
    """Convert SQLAlchemy Job object to dictionary matching the pipeline expectations."""
    return {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "remote": job.remote,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "contract_type": job.contract_type,
        "description": job.description,
        "skills_required": job.skills_required,
        "source": job.source,
        "posted_at": job.posted_at,
        "url": job.url,
        "seniority": None,  # Will be extracted by ensure_job_metadata
        "industry": None,
        "vector": job.embedding,
    }


SENIORITY_ORDER = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
}


def normalize_skill_set(values):
    return {value.lower() for value in normalize_list(values)}


def tokenize(text):
    return set(re.findall(r"[a-z0-9#+.]+", (text or "").lower()))


def weighted_average(parts):
    valid_parts = [(score, weight) for score, weight in parts if score is not None]
    total_weight = sum(weight for _, weight in valid_parts)

    if total_weight == 0:
        return 0.0

    return sum(score * weight for score, weight in valid_parts) / total_weight


def ensure_job_metadata(job):
    metadata = extract_job_metadata(job.get("title"), job.get("description"))

    return {
        **job,
        "skills_required": job.get("skills_required") or metadata["skills_required"],
        "seniority": job.get("seniority") or metadata["seniority"],
        "industry": job.get("industry") or metadata["industry"],
    }


def compute_position_score(target_position, job):
    if not target_position:
        return 0.0

    target_tokens = tokenize(target_position)
    title_tokens = tokenize(job.get("title"))
    description_tokens = tokenize(job.get("description"))
    overlap_with_title = len(target_tokens & title_tokens) / max(len(target_tokens), 1)
    overlap_with_description = len(target_tokens & description_tokens) / max(len(target_tokens), 1)
    exact_match = 1.0 if target_position.lower() in build_job_text(job).lower() else 0.0

    return min(1.0, (0.6 * overlap_with_title) + (0.2 * overlap_with_description) + (0.2 * exact_match))


def compute_skill_score(profile_skills, job_skills):
    if not profile_skills:
        return None

    job_skill_set = normalize_skill_set(job_skills)
    if not job_skill_set:
        return 0.0

    return len(profile_skills & job_skill_set) / len(profile_skills)


def compute_seniority_score(profile_seniority, job_seniority):
    if not profile_seniority:
        return None

    if not job_seniority:
        return 0.0

    profile_rank = SENIORITY_ORDER.get(str(profile_seniority).lower())
    job_rank = SENIORITY_ORDER.get(str(job_seniority).lower())
    if profile_rank is None or job_rank is None:
        return 0.0

    difference = abs(profile_rank - job_rank)
    return max(0.0, 1.0 - (difference / len(SENIORITY_ORDER)))


def compute_industry_score(profile_industries, job_industry):
    if not profile_industries:
        return None

    if not job_industry:
        return 0.0

    return 1.0 if job_industry.lower() in profile_industries else 0.0


def compute_recency_score(posted_at):
    if not posted_at:
        return 0.0

    try:
        parsed = datetime.fromisoformat(str(posted_at).replace("Z", "+00:00"))
    except ValueError:
        return 0.0

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    age_in_days = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 86400)
    return max(0.0, 1.0 - min(age_in_days, 30) / 30)


def compute_proximity_score(profile, job):
    remote_preference = str(profile.get("remote_preference", "any")).lower()
    profile_location = profile.get("location")
    job_location = job.get("location")

    if job.get("remote"):
        return 1.0 if remote_preference in {"any", "remote", "hybrid"} else 0.6

    if not profile_location or not job_location:
        return 0.5

    profile_tokens = tokenize(profile_location)
    job_tokens = tokenize(job_location)
    if not profile_tokens or not job_tokens:
        return 0.3

    overlap = len(profile_tokens & job_tokens) / len(profile_tokens)
    if overlap > 0:
        return min(1.0, 0.4 + (0.6 * overlap))

    return 0.1 if remote_preference == "remote" else 0.2


def keyword_filter_jobs(profile, jobs, limit=40):
    profile_skills = normalize_skill_set(profile.get("skills"))
    profile_industries = {value.lower() for value in normalize_list(profile.get("industries") or profile.get("industry"))}
    profile_level = profile.get("user_level") or profile.get("seniority")
    filtered_jobs = []

    for job in jobs:
        position_score = compute_position_score(profile.get("target_position"), job)
        skill_score = compute_skill_score(profile_skills, job.get("skills_required"))
        seniority_score = compute_seniority_score(profile_level, job.get("seniority"))
        industry_score = compute_industry_score(profile_industries, job.get("industry"))
        matched_skills = sorted(profile_skills & normalize_skill_set(job.get("skills_required")))
        keyword_score = weighted_average([
            (position_score, 0.45),
            (skill_score, 0.35),
            (seniority_score, 0.10),
            (industry_score, 0.10),
        ])

        filtered_jobs.append({
            **job,
            "matched_skills": matched_skills,
            "keyword_score": round(keyword_score, 4),
            "position_score": round(position_score, 4),
            "skill_score": round(skill_score or 0.0, 4),
            "seniority_score": round(seniority_score or 0.0, 4),
            "industry_score": round(industry_score or 0.0, 4),
        })

    filtered_jobs.sort(key=lambda item: item["keyword_score"], reverse=True)
    return filtered_jobs[:limit]


def _profile_embedding_cache_key(profile):
    profile_text = build_profile_text(profile)
    digest = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    return f"profile_embedding:{digest}", profile_text


def get_or_create_profile_embedding(profile):
    cache = get_cache()
    profile_cache_key, profile_text = _profile_embedding_cache_key(profile)
    profile_embedding = profile.get("vector") or cache.get_json(profile_cache_key)
    if profile_embedding is None:
        profile_embedding = embed(profile_text)
        cache.set_json(profile_cache_key, profile_embedding, PROFILE_EMBED_TTL_SECONDS)
    return profile_embedding


def semantic_rank_jobs(profile_embedding, jobs, limit=25):
    if not jobs:
        return [], {"job_vectors_from_store": 0, "job_vectors_embedded": 0, "embedding_ms": 0, "similarity_ms": 0}

    embedding_started = perf_counter()
    jobs_to_embed = [job for job in jobs if not job.get("vector")]
    from_store = len(jobs) - len(jobs_to_embed)
    embedded_jobs = embed_many([build_job_text(job) for job in jobs_to_embed]) if jobs_to_embed else []
    embedding_elapsed_ms = int((perf_counter() - embedding_started) * 1000)

    embedded_by_id = {
        job.get("id"): job_embedding
        for job, job_embedding in zip(jobs_to_embed, embedded_jobs)
    }
    similarity_started = perf_counter()
    vectors = []
    jobs_with_vectors = []
    for job in jobs:
        job_embedding = job.get("vector") or embedded_by_id.get(job.get("id"))
        if job_embedding is None:
            continue
        jobs_with_vectors.append(job)
        vectors.append(job_embedding)

    indices, scores = cosine_top_k(profile_embedding, vectors, top_k=limit)
    ranked_jobs = []
    for idx, score in zip(indices.tolist(), scores.tolist()):
        job = jobs_with_vectors[idx]
        ranked_jobs.append({
            **job,
            "semantic_similarity": round(float(score), 4),
            "vector": job.get("vector") or vectors[idx],
        })

    similarity_elapsed_ms = int((perf_counter() - similarity_started) * 1000)
    stats = {
        "job_vectors_from_store": from_store,
        "job_vectors_embedded": len(jobs_to_embed),
        "embedding_ms": embedding_elapsed_ms,
        "similarity_ms": similarity_elapsed_ms,
    }
    return ranked_jobs[:limit], stats


async def retrieve_candidates(profile, db=None, jobs=None, country="fr", fetch_limit=50, keyword_limit=40, similarity_limit=25):
    retrieval_started_at = perf_counter()
    stage_timers = {}
    target_position = profile.get("target_position")
    if not target_position:
        raise ValueError("Missing profile target_position")

    fetch_started = perf_counter()
    if jobs is None:
        raw_jobs, source_counts = await fetch_jobs_hybrid_async(
            db, target_position, country, fetch_limit, profile.get("location")
        )
    else:
        raw_jobs = jobs[:fetch_limit]
        source_counts = {"db": 0, "external": len(raw_jobs), "db_ms": 0, "external_ms": 0}
    stage_timers["fetch_ms"] = int((perf_counter() - fetch_started) * 1000)

    def _normalize_one(raw_job):
        mapped_job = map_adzuna_job(raw_job) if raw_job.get("redirect_url") else raw_job
        return ensure_job_metadata(mapped_job)

    normalize_started = perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        normalized_jobs = list(executor.map(_normalize_one, raw_jobs))
    stage_timers["normalize_ms"] = int((perf_counter() - normalize_started) * 1000)

    profile_embedding_started = perf_counter()
    profile_embedding = get_or_create_profile_embedding(profile)
    stage_timers["profile_embedding_ms"] = int((perf_counter() - profile_embedding_started) * 1000)

    ann_started = perf_counter()
    ann_jobs = query_ann_jobs(
        db=db,
        profile_embedding=profile_embedding,
        top_k=keyword_limit,
    ) if db is not None else []
    stage_timers["ann_ms"] = int((perf_counter() - ann_started) * 1000)
    candidate_pool = ann_jobs if ann_jobs else normalized_jobs

    keyword_started = perf_counter()
    keyword_ranked_jobs = keyword_filter_jobs(profile, candidate_pool, limit=keyword_limit)
    stage_timers["keyword_ms"] = int((perf_counter() - keyword_started) * 1000)

    semantic_started = perf_counter()
    similarity_ranked_jobs, vector_stats = semantic_rank_jobs(profile_embedding, keyword_ranked_jobs, limit=similarity_limit)
    stage_timers["semantic_total_ms"] = int((perf_counter() - semantic_started) * 1000)

    retrieval_elapsed_ms = int((perf_counter() - retrieval_started_at) * 1000)
    retrieval_breakdown = {
        "mode": "hybrid",
        "stages": {
            "fetched": len(raw_jobs),
            "normalized": len(normalized_jobs),
            "ann_candidates": len(ann_jobs),
            "keyword_filtered": len(keyword_ranked_jobs),
            "semantic_ranked": len(similarity_ranked_jobs),
            "source_db": source_counts["db"],
            "source_external": source_counts["external"],
        },
        "vectors": vector_stats,
        "timers_ms": stage_timers,
        "source_timers_ms": {
            "db_query": source_counts.get("db_ms", 0),
            "external_fetch": source_counts.get("external_ms", 0),
        },
        "latency_ms": retrieval_elapsed_ms,
    }

    return similarity_ranked_jobs, retrieval_breakdown


def _build_explanations(job):
    explanations = []
    for reason in job.get("match_summary", []):
        if reason:
            explanations.append(str(reason))

    if job.get("matched_skills"):
        explanations.append(f"Skills overlap: {', '.join(job['matched_skills'][:5])}")

    if job.get("semantic_similarity") is not None:
        explanations.append(f"Semantic fit: {round(float(job['semantic_similarity']) * 100)}%")

    if job.get("keyword_score") is not None:
        explanations.append(f"Keyword fit: {round(float(job['keyword_score']) * 100)}%")

    return explanations[:5]


def rerank_candidates(profile, candidates, final_limit=10):
    rerank_started_at = perf_counter()

    enriched_jobs = []
    for job in candidates:
        enriched_jobs.append({
            **job,
            "recency_score": round(compute_recency_score(job.get("posted_at")), 4),
            "proximity_score": round(compute_proximity_score(profile, job), 4),
        })

    final_jobs = rerank_jobs(profile, enriched_jobs, limit=final_limit)
    rerank_elapsed_ms = int((perf_counter() - rerank_started_at) * 1000)

    explained_jobs = []
    for job in final_jobs:
        explained_jobs.append({
            **job,
            "explanations": _build_explanations(job),
        })

    return explained_jobs, rerank_elapsed_ms


def _result_cache_key(profile, country, fetch_limit, keyword_limit, similarity_limit, final_limit):
    payload = {
        "profile_text": build_profile_text(profile),
        "country": country,
        "fetch_limit": fetch_limit,
        "keyword_limit": keyword_limit,
        "similarity_limit": similarity_limit,
        "final_limit": final_limit,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"match_result:{digest}"


async def run_matching_pipeline(profile, db=None, jobs=None, country="fr", fetch_limit=None, keyword_limit=None, similarity_limit=None, final_limit=None):
    fetch_limit = fetch_limit or FETCH_LIMIT
    keyword_limit = keyword_limit or KEYWORD_TOP_K
    similarity_limit = similarity_limit or SIMILARITY_TOP_K
    final_limit = final_limit or FINAL_TOP_K
    pipeline_started_at = perf_counter()
    cache = get_cache()
    cache_key = _result_cache_key(profile, country, fetch_limit, keyword_limit, similarity_limit, final_limit)
    cached_response = cache.get_json(cache_key)
    if cached_response is not None:
        cached_response["cache_hit"] = True
        return cached_response

    candidates, retrieval_breakdown = await retrieve_candidates(
        profile=profile,
        db=db,
        jobs=jobs,
        country=country,
        fetch_limit=fetch_limit,
        keyword_limit=keyword_limit,
        similarity_limit=similarity_limit,
    )
    elapsed_before_rerank_ms = int((perf_counter() - pipeline_started_at) * 1000)
    degraded_mode = elapsed_before_rerank_ms > REQUEST_TIMEOUT_MS
    if degraded_mode:
        final_jobs = candidates[:final_limit]
        rerank_latency_ms = 0
    else:
        final_jobs, rerank_latency_ms = rerank_candidates(profile, candidates, final_limit=final_limit)
    total_latency_ms = int((perf_counter() - pipeline_started_at) * 1000)
    response = {
        "query": profile.get("target_position"),
        "country": country,
        "cache_hit": False,
        "degraded_mode": degraded_mode,
        "stage_counts": {
            "fetched": retrieval_breakdown["stages"]["fetched"],
            "normalized": retrieval_breakdown["stages"]["normalized"],
            "ann_candidates": retrieval_breakdown["stages"]["ann_candidates"],
            "keyword_filtered": retrieval_breakdown["stages"]["keyword_filtered"],
            "semantic_ranked": retrieval_breakdown["stages"]["semantic_ranked"],
            "final": len(final_jobs),
        },
        "latency_ms": {
            "total": total_latency_ms,
            "retrieval": retrieval_breakdown["latency_ms"],
            "db_query": retrieval_breakdown["source_timers_ms"].get("db_query", 0),
            "external_fetch": retrieval_breakdown["source_timers_ms"].get("external_fetch", 0),
            "keyword_filtering": retrieval_breakdown["timers_ms"].get("keyword_ms", 0),
            "embedding": retrieval_breakdown["vectors"].get("embedding_ms", 0),
            "similarity_search": retrieval_breakdown["vectors"].get("similarity_ms", 0),
            "ann": retrieval_breakdown["timers_ms"].get("ann_ms", 0),
            "rerank": rerank_latency_ms,
        },
        "retrieval_breakdown": retrieval_breakdown,
        "jobs": final_jobs,
    }
    cache.set_json(cache_key, response, RESULT_CACHE_TTL_SECONDS)
    return response
