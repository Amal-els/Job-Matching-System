import time
import numpy as np

from models.job import Job

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency fallback
    faiss = None


_INDEX = None
_JOB_IDS = []
_INDEXED_AT = 0.0
INDEX_TTL_SECONDS = 300


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _build_index(db):
    global _INDEX, _JOB_IDS, _INDEXED_AT
    if faiss is None or db is None:
        return

    rows = db.query(Job.id, Job.embedding).filter(Job.embedding.isnot(None)).all()
    vectors = []
    job_ids = []
    for row in rows:
        embedding = row.embedding
        if not embedding:
            continue
        vectors.append(np.array(embedding, dtype="float32"))
        job_ids.append(row.id)

    if not vectors:
        _INDEX = None
        _JOB_IDS = []
        _INDEXED_AT = time.time()
        return

    matrix = np.vstack(vectors).astype("float32")
    matrix = _normalize(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    _INDEX = index
    _JOB_IDS = job_ids
    _INDEXED_AT = time.time()


def query_ann_jobs(db, profile_embedding, top_k=30):
    if faiss is None or db is None or profile_embedding is None:
        return []

    should_rebuild = _INDEX is None or (time.time() - _INDEXED_AT) > INDEX_TTL_SECONDS
    if should_rebuild:
        _build_index(db)
    if _INDEX is None or not _JOB_IDS:
        return []

    profile = np.array(profile_embedding, dtype="float32").reshape(1, -1)
    profile = _normalize(profile)
    _, indices = _INDEX.search(profile, int(top_k))

    selected_ids = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(_JOB_IDS):
            continue
        selected_ids.append(_JOB_IDS[idx])

    if not selected_ids:
        return []

    jobs = db.query(Job).filter(Job.id.in_(selected_ids)).all()
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
            "seniority": None,
            "industry": None,
            "vector": job.embedding,
        })
    return results
