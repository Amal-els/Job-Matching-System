# 🔬 CareerMate AI: Technical Deep Dive

This document provides an exhaustive, engineer-level explanation of every major component in the CareerMate AI ecosystem and how they interact to provide high-frequency job matching.

---

## 1. The Orchestration Layer (`ai-service/routes/`)

The system uses [FastAPI](https://fastapi.tiangolo.com/) for its speed and native `asyncio` support.

### `match.py`: The Pipeline Conductor
This is the most critical entry point. When you call `POST /match/`, the following happens:
1.  **Validation**: Ensures the profile exists or is correctly shaped in the payload.
2.  **Concurrency**: It launches a `Hybrid Fetch` (looking in the DB and calling the Adzuna API) concurrently to minimize wait time.
3.  **Latency Budgeting**: It checks a stopwatch. If the retrieval took > 800ms, it flags the request for "Degraded Mode" to skip heavy LLM processing.

---

## 2. The Intelligence Logic (`ai-service/services/`)

This layer is where the "AI" part happens. It is stateless and relies on the database for persistence.

### `matching.py`: The Multi-Stage Funnel
We use a narrowing approach because embedding and comparing 10,000 jobs is too slow for real-time.
-   **Stage 1: ANN Retrieval**: Uses `pgvector` to find jobs that are "mathematically close" in vector space.
-   **Stage 2: Keyword Scoring**: Uses specific weights:
    -   **Position (45%)**: Title-to-Target overlap.
    -   **Skills (35%)**: Hard skill set intersection.
    -   **Seniority (10%)**: Difference between 'Junior', 'Mid', 'Senior' ranks.
    -   **Industry (10%)**: Domain vertical match.

### `embedding.py`: Vectorization
This service translates human language into a list of 384 numbers.
-   **Provider Fallback**: It tries a dedicated embedding microservice first. If that's down, it can fallback to external APIs or a local deterministic transformer.
-   **Stability**: It uses a centralized `SharedTextBuilder` so that a "Python Backend Dev" always generates the exact same vector, ensuring cache hits.

---

## 3. Data & Persistence (`ai-service/models/` & `database.py`)

### `pgvector`: The Vector Database
Unlike standard Postgres, we use the `pgvector` extension. This allows us to run `ORDER BY vector <=> {query_vector}` which is a high-speed cosine distance calculation performed inside the database engine.

### `ingestion/`: The Data Harvester
-   **`adzuna.py`**: Handles API rate limits and country-specific fetching.
-   **`mapper.py`**: Normalizes "Software Engineer II" into "Software Engineer" and "mid-level" using deterministic rules and LLM assistance where needed.

---

## 4. Resilience & Performance

### `cache.py`: Dual-Layer Caching
1.  **Embedding Cache**: Saves us from re-embedding the same user bio repeatedly.
2.  **Result Cache**: If a user refreshes the page with the same profile, we serve the results from Redis in < 10ms.

### Stale Job Cleanup
The system isn't just about adding data; it's about pruning. In `matching.py`, the `cleanup_stale_jobs` function runs regularly to remove jobs older than 14 days, keeping the indexes dense and fast.

---

## 5. Development Utilities

-   **`migrate.py`**: Safe migrations that ensure the `vector` extension is enabled in Postgres before trying to create tables.
-   **`backfill_job_embeddings.py`**: A utility for "lazy" migration. If you import 10,000 jobs without vectors, this script will churn through them in the background.

---

## 🛠 Summary of Flow
1. **Input**: User Profile.
2. **Retrieve**: DB (ANN) + Adzuna (API).
3. **Persist**: New jobs are saved & embedded in the background.
4. **Rank**: Keyword Scoring -> Semantic Scoring -> Reranking.
5. **Output**: Top 10 jobs with "Explanations" (e.g., "90% skill overlap").
