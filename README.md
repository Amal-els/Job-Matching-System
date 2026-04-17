# Mini Job Matcher

Lightweight AI-powered job matching service built with FastAPI.  
The service ingests jobs, enriches metadata, computes embeddings, and returns ranked matches for a user profile.

## Current Status

The system is functional and currently focused on latency improvements.  
Recent work has already introduced a staged matching pipeline and reduced unnecessary embedding calls at request time.

## Latency Roadmap Progress

- Phase 1 (Measure): Implemented
- Phase 2 (Reduce Work Early): Implemented with top-K defaults
- Phase 3 (ANN via FAISS): Implemented with safe fallback when FAISS is unavailable
- Phase 4 (Caching): Implemented with Redis support and in-memory fallback
- Phase 5 (Async Pipeline): Implemented for `/match` with concurrent DB + external fetch
- Phase 6 (Timeout Fallback): Implemented via request latency budget

## Architecture Overview

- `ai-service/main.py`: FastAPI app bootstrap and route registration.
- `ai-service/routes/match.py`: `POST /match/` endpoint entry point.
- `ai-service/services/matching.py`: end-to-end retrieval + ranking pipeline.
- `ai-service/services/job_text.py`: single source of truth for profile/job text building.
- `ai-service/services/embedding.py`: embedding client that calls `embedding-service` first, then falls back.
- `embedding-service/main.py`: dedicated embedding microservice with in-memory model.
- `ai-service/models/job.py`: `jobs` table model including persisted embedding.
- `ai-service/migrate.py`: schema migration for `posted_at`, `url`, `created_at`, and `embedding`.

## Matching Pipeline (Current)

1. **Hybrid retrieval**
   - Search fresh jobs from DB first.
   - Fetch missing jobs from Adzuna when DB results are insufficient.
2. **Normalization and metadata enrichment**
   - Parse/derive skills, seniority, and industry where missing.
3. **Keyword filtering**
   - Score title/description overlap, skills, seniority, and industry.
4. **Semantic ranking**
   - Compare profile embedding against job embeddings.
5. **Final reranking**
   - Add recency and proximity signals, then rerank and attach explanations.

## Latency Work Already Implemented

- **Precomputed job embeddings at ingest/save time**
  - New jobs are embedded once and stored in DB (`jobs.embedding`).
  - Match requests reuse stored vectors when available.
- **Shared text builders**
  - `build_profile_text` and `build_job_text` are centralized to keep embedding input stable.
- **Batch embedding support**
  - `embed_many()` allows grouped calls instead of per-item overhead.
- **Staged narrowing**
  - Candidate set shrinks from fetched -> keyword filtered -> semantic ranked -> final.
- **Per-stage latency reporting**
  - Response includes stage-specific timings: DB query, external fetch, keyword filtering, embedding, similarity search, ANN, rerank.
  - Response includes candidate counts at each stage.
- **FAISS ANN retrieval**
  - Uses ANN top-K retrieval from persisted job embeddings when available.
  - Falls back to the existing pipeline if FAISS is missing or index is unavailable.
- **Caching layer**
  - Profile embedding cache (TTL 1 hour by default).
  - Final result cache (TTL 10 minutes by default).
  - Uses Redis when `REDIS_URL` is configured; otherwise falls back to in-memory cache.
- **Async request path**
  - `/match/` is async and concurrently triggers DB and external fetch steps.
- **Timeout fallback**
  - If retrieval exceeds request budget, rerank is skipped and best available candidates are returned.
- **Resilience fallback**
  - If remote embedding provider is unavailable, a deterministic local embedding fallback keeps the service responsive.
- **Separate embedding service**
  - Embeddings can be served by a dedicated microservice (`/embed`, `/embed-many`) with one-time model load at startup.
  - Main API remains responsive even if embedding workload grows; service can scale independently.

## API

### Health

- `GET /` -> `{"status":"ok"}`

### Match

- `POST /match/`
- Accepts either `user` or `profile` in request body.
- `target_position` is required.

Example payload:

```json
{
  "user": {
    "id": "u-1",
    "target_position": "Backend Developer",
    "skills": ["Python", "FastAPI", "SQL"],
    "location": "Paris",
    "user_level": "mid"
  },
  "country": "fr",
  "fetch_limit": 50,
  "keyword_limit": 40,
  "similarity_limit": 25,
  "final_limit": 10
}
```

Example response fields (trimmed):

- `query`
- `stage_counts`
- `latency_ms` (`total`, `retrieval`, `rerank`)
- `retrieval_breakdown`
- `jobs` (ranked results + explanations)

## Setup

## 1) Install dependencies

```bash
cd ai-service
pip install -r requirements.txt
```

## 2) Configure environment

Typical variables:

- `DATABASE_URL`
- `REDIS_URL` (optional, enables Redis cache)
- `EMBEDDING_SERVICE_URL` (default docker value: `http://embedding-service:8001`)
- `HF_TOKEN` (for Hugging Face embedding provider)
- `EMBEDDING_MODEL_NAME` (optional, default: `sentence-transformers/all-MiniLM-L6-v2`)
- `EMBEDDING_FALLBACK_DIM` (optional, default: `384`)
- `FETCH_LIMIT` (default: `100`)
- `KEYWORD_TOP_K` (default: `30`)
- `SIMILARITY_TOP_K` (default: `15`)
- `FINAL_TOP_K` (default: `10`)
- `REQUEST_TIMEOUT_MS` (default: `800`)
- `PROFILE_EMBED_TTL_SECONDS` (default: `3600`)
- `RESULT_CACHE_TTL_SECONDS` (default: `600`)
- `JOB_TTL_DAYS` (default: `14`, stale jobs older than TTL are removed from DB)
- Adzuna credentials used by ingestion layer (if enabled in your environment)

## 3) Run migration

```bash
python migrate.py
```

## 4) Start API

```bash
uvicorn main:app --reload
```

Using Docker Compose (recommended for Redis + embedding service):

```bash
docker compose up --build
```

## Backfill Existing Job Embeddings

If your DB already contains jobs created before embedding persistence, run:

```bash
cd ai-service
python backfill_job_embeddings.py
```

This populates `jobs.embedding` for rows where it is currently `NULL`, which enables stronger FAISS ANN retrieval.

## Embedding Service 

The project now supports a separate embedding microservice:

- `POST /embed` with `{ "text": "..." }`
- `POST /embed-many` with `{ "texts": ["...", "..."] }`

When `EMBEDDING_SERVICE_URL` is set, `ai-service` calls this service first for embeddings.
If unavailable, it falls back to existing providers/fallback vectors.

## Next Latency Improvements (Planned)

These are practical production-style optimizations to consider next:

- Add request-level caching for repeated profile queries.
- Move toward ANN/vector-index retrieval (instead of broad scan + rerank).
- Add timeout-based graceful degradation for slow external providers.
- Profile and parallelize high-cost sections with strict p95/p99 targets.
- Introduce async I/O for external APIs and heavy network-bound operations.

## Notes

- Current repo includes generated `__pycache__` artifacts; they should be excluded from commits in normal workflows.
- `requirements.txt` currently contains duplicate entries (`requests` appears twice). Cleanup can be done in a follow-up.
