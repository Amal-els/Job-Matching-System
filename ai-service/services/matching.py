import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

from sqlalchemy import or_, and_
from models.job import Job
from ingestion.adzuna import fetch_adzuna_jobs
from ingestion.mapper import map_adzuna_job
from services.embedding import embed, embed_many
from services.job_parser import extract_job_metadata
from services.reranker import rerank_jobs
from services.similarity import cosine


def fetch_jobs_hybrid(db, query, country="fr", limit=50, location=None):
    """
    Search DB first for fresh jobs, then enrich from Adzuna if needed.
    """
    if db is None:
        return fetch_adzuna_jobs(country=country, query=query, limit=limit, location=location)

    # 1. Search DB for fresh matches (last 48 hours)
    threshold = datetime.now(timezone.utc) - timedelta(hours=48)
    
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
    """
    if not db or not jobs:
        return

    for job_data in jobs:
        # Avoid duplicate primary key error
        exists = db.query(Job).filter(Job.id == job_data["id"]).first()
        if exists:
            continue

        job = Job(
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
            url=job_data.get("url")
        )
        db.add(job)
    
    try:
        db.commit()
    except Exception:
        db.rollback()


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
        "vector": None,
    }


SENIORITY_ORDER = {
    "intern": 0,
    "junior": 1,
    "mid": 2,
    "senior": 3,
    "lead": 4,
}


def _scalar_to_text(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip()


def flatten_text(value):
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return _scalar_to_text(value)

    if isinstance(value, dict):
        preferred_keys = [
            "name", "title", "label", "value", "skill", "role", "position",
            "company", "school", "degree", "language", "certificate", "certification",
            "description", "summary",
        ]
        preferred_parts = [
            _scalar_to_text(value.get(key))
            for key in preferred_keys
            if value.get(key) not in (None, "", [], {})
        ]
        if preferred_parts:
            return " | ".join(part for part in preferred_parts if part)

        parts = []
        for key, item in value.items():
            flattened = flatten_text(item)
            if flattened:
                parts.append(f"{key}: {flattened}")
        return " | ".join(parts)

    if isinstance(value, (list, tuple, set)):
        return " ; ".join(part for part in (flatten_text(item) for item in value) if part)

    return str(value)


def normalize_list(values):
    if not values:
        return []

    if isinstance(values, (str, int, float, bool, dict)):
        values = [values]

    normalized = []
    for value in values:
        flattened = flatten_text(value).strip()
        if flattened:
            normalized.append(flattened)

    return normalized


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


def build_profile_text(profile):
    parts = [
        f"Profile ID: {profile.get('id', '')}",
        f"Target position: {profile.get('target_position', '')}",
        f"Bio: {profile.get('bio', '')}",
        f"User level: {profile.get('user_level', '')}",
        f"Profile score: {profile.get('profil_score', '')}",
        f"Skills: {', '.join(normalize_list(profile.get('skills')))}",
        f"Experiences: {flatten_text(profile.get('experiences'))}",
        f"Education: {flatten_text(profile.get('education'))}",
        f"Languages: {', '.join(normalize_list(profile.get('languages')))}",
        f"Certifications: {', '.join(normalize_list(profile.get('certifications')))}",
        f"Short term goals: {flatten_text(profile.get('short_term_goals'))}",
        f"Long term goals: {flatten_text(profile.get('long_term_goals'))}",
        f"Location: {profile.get('location', '')}",
    ]

    return "\n".join(part for part in parts if part.split(":", 1)[1].strip())


def build_job_text(job):
    parts = [
        f"Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Description: {job.get('description', '')}",
        f"Skills: {', '.join(normalize_list(job.get('skills_required')))}",
        f"Seniority: {job.get('seniority', '')}",
        f"Industry: {job.get('industry', '')}",
    ]

    return "\n".join(part for part in parts if part.split(":", 1)[1].strip())


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


def semantic_rank_jobs(profile, jobs, limit=25):
    if not jobs:
        return []

    profile_embedding = profile.get("vector") or embed(build_profile_text(profile))
    jobs_to_embed = [job for job in jobs if not job.get("vector")]
    embedded_jobs = embed_many([build_job_text(job) for job in jobs_to_embed]) if jobs_to_embed else []
    embedded_by_id = {
        job.get("id"): job_embedding
        for job, job_embedding in zip(jobs_to_embed, embedded_jobs)
    }
    ranked_jobs = []

    for job in jobs:
        job_embedding = job.get("vector") or embedded_by_id.get(job.get("id"))
        if job_embedding is None:
            continue

        similarity_score = cosine(profile_embedding, job_embedding)
        ranked_jobs.append({
            **job,
            "semantic_similarity": round(similarity_score, 4),
            "vector": job.get("vector") or job_embedding,
        })

    ranked_jobs.sort(key=lambda item: item["semantic_similarity"], reverse=True)
    return ranked_jobs[:limit]


def run_matching_pipeline(profile, db=None, jobs=None, country="fr", fetch_limit=50, keyword_limit=40, similarity_limit=25, final_limit=10):
    target_position = profile.get("target_position")
    if not target_position:
        raise ValueError("Missing profile target_position")

    if jobs is None:
        raw_jobs = fetch_jobs_hybrid(
            db=db,
            country=country,
            query=target_position,
            limit=fetch_limit,
            location=profile.get("location"),
        )
    else:
        raw_jobs = jobs[:fetch_limit]

    def _normalize_one(raw_job):
        mapped_job = map_adzuna_job(raw_job) if raw_job.get("redirect_url") else raw_job
        return ensure_job_metadata(mapped_job)

    with ThreadPoolExecutor(max_workers=8) as executor:
        normalized_jobs = list(executor.map(_normalize_one, raw_jobs))

    keyword_ranked_jobs = keyword_filter_jobs(profile, normalized_jobs, limit=keyword_limit)
    similarity_ranked_jobs = semantic_rank_jobs(profile, keyword_ranked_jobs, limit=similarity_limit)

    enriched_jobs = []
    for job in similarity_ranked_jobs:
        enriched_jobs.append({
            **job,
            "recency_score": round(compute_recency_score(job.get("posted_at")), 4),
            "proximity_score": round(compute_proximity_score(profile, job), 4),
        })

    final_jobs = rerank_jobs(profile, enriched_jobs, limit=final_limit)

    return {
        "query": target_position,
        "country": country,
        "stage_counts": {
            "fetched": len(raw_jobs),
            "keyword_filtered": len(keyword_ranked_jobs),
            "semantic_ranked": len(similarity_ranked_jobs),
            "final": len(final_jobs),
        },
        "jobs": final_jobs,
    }
