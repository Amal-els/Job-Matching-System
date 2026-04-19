import os
import time
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import func

from models.job import Job

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency fallback
    faiss = None


_INDEX = None
_JOB_IDS = []
_INDEXED_AT = 0.0
INDEX_JOB_TTL_DAYS = int(os.getenv("JOB_TTL_DAYS", "14"))
INDEX_STATE_CHECK_SECONDS = max(1, int(os.getenv("ANN_INDEX_STATE_CHECK_SECONDS", "30")))
_INDEX_SOURCE_STATE = None
_INDEX_STATE_CHECKED_AT = 0.0
_INDEX_DIRTY = True
_INDEX_LOCK = threading.Lock()


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _index_threshold():
    return datetime.now(timezone.utc) - timedelta(days=INDEX_JOB_TTL_DAYS)


def _current_source_state(db):
    threshold = _index_threshold()
    count_rows, latest_created_at = db.query(
        func.count(Job.id),
        func.max(Job.created_at),
    ).filter(
        Job.vector.isnot(None),
        Job.created_at >= threshold,
    ).one()
    latest_token = latest_created_at.isoformat() if latest_created_at else None
    return int(count_rows or 0), latest_token


def mark_ann_index_dirty():
    global _INDEX_DIRTY
    with _INDEX_LOCK:
        _INDEX_DIRTY = True


def _should_refresh_source_state(db):
    global _INDEX_STATE_CHECKED_AT, _INDEX_SOURCE_STATE

    now = time.time()
    if _INDEX_DIRTY or _INDEX_SOURCE_STATE is None:
        _INDEX_SOURCE_STATE = _current_source_state(db)
        _INDEX_STATE_CHECKED_AT = now
        return True

    if (now - _INDEX_STATE_CHECKED_AT) < INDEX_STATE_CHECK_SECONDS:
        return False

    current_state = _current_source_state(db)
    _INDEX_STATE_CHECKED_AT = now
    if current_state != _INDEX_SOURCE_STATE:
        _INDEX_SOURCE_STATE = current_state
        return True
    return False


def _build_index(db):
    global _INDEX, _JOB_IDS, _INDEXED_AT, _INDEX_SOURCE_STATE, _INDEX_STATE_CHECKED_AT, _INDEX_DIRTY
    if faiss is None or db is None:
        return

    threshold = _index_threshold()
    rows = db.query(Job.id, Job.vector).filter(
        Job.vector.isnot(None),
        Job.created_at >= threshold,
    ).all()
    vectors = []
    job_ids = []
    for row in rows:
        embedding = row.vector
        if not embedding:
            continue
        vectors.append(np.array(embedding, dtype="float32"))
        job_ids.append(row.id)

    if not vectors:
        _INDEX = None
        _JOB_IDS = []
        _INDEXED_AT = time.time()
        _INDEX_SOURCE_STATE = (0, None)
        _INDEX_STATE_CHECKED_AT = _INDEXED_AT
        _INDEX_DIRTY = False
        return

    matrix = np.vstack(vectors).astype("float32")
    matrix = _normalize(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    _INDEX = index
    _JOB_IDS = job_ids
    _INDEXED_AT = time.time()
    _INDEX_SOURCE_STATE = _current_source_state(db)
    _INDEX_STATE_CHECKED_AT = _INDEXED_AT
    _INDEX_DIRTY = False


def query_ann_jobs(db, profile_embedding, top_k=30):
    if faiss is None or db is None or profile_embedding is None:
        return []

    with _INDEX_LOCK:
        if _should_refresh_source_state(db):
            _build_index(db)
        if _INDEX is None or not _JOB_IDS:
            return []
        index = _INDEX
        job_ids = list(_JOB_IDS)

    profile = np.array(profile_embedding, dtype="float32").reshape(1, -1)
    profile = _normalize(profile)
    _, indices = index.search(profile, int(top_k))

    selected_ids = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(job_ids):
            continue
        selected_ids.append(job_ids[idx])

    if not selected_ids:
        return []

    threshold = _index_threshold()
    jobs = db.query(Job).filter(
        Job.id.in_(selected_ids),
        Job.created_at >= threshold,
    ).all()
    job_by_id = {job.id: job for job in jobs}
    ordered = [job_by_id[job_id] for job_id in selected_ids if job_id in job_by_id]

    results = []
    for job in ordered:
        results.append({
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
            "seniority": job.seniority,
            "industry": job.industry,
            "vector": job.vector,
        })
    return results
