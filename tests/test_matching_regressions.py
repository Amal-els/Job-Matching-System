import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch


os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.append(str(Path(__file__).resolve().parents[1] / "ai-service"))

from services import matching  # noqa: E402


class FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def one(self):
        return self._result


class FakeDB:
    def __init__(self, result):
        self._result = result

    def query(self, *args, **kwargs):
        return FakeQuery(self._result)


class MatchingCacheKeyTests(unittest.TestCase):
    def test_cache_key_changes_when_jobs_payload_changes(self):
        profile = {"target_position": "Data Scientist"}

        key_one = matching._result_cache_key(
            profile,
            "fr",
            50,
            40,
            25,
            10,
            jobs=[{"id": "job-1", "title": "Backend Engineer"}],
            db=None,
        )
        key_two = matching._result_cache_key(
            profile,
            "fr",
            50,
            40,
            25,
            10,
            jobs=[{"id": "job-2", "title": "Data Scientist"}],
            db=None,
        )

        self.assertNotEqual(key_one, key_two)

    def test_cache_key_changes_when_job_store_state_changes(self):
        profile = {"target_position": "Data Scientist"}
        older_state = FakeDB((10, datetime(2026, 4, 18, tzinfo=timezone.utc)))
        newer_state = FakeDB((11, datetime(2026, 4, 19, tzinfo=timezone.utc)))

        key_one = matching._result_cache_key(profile, "fr", 50, 40, 25, 10, db=older_state)
        key_two = matching._result_cache_key(profile, "fr", 50, 40, 25, 10, db=newer_state)

        self.assertNotEqual(key_one, key_two)


class RetrieveCandidatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_candidates_still_fetches_hybrid_jobs_when_ann_is_full(self):
        profile = {"target_position": "Data Scientist", "location": "Tunis"}
        ann_jobs = [
            {"id": f"ann-{index}", "title": f"Old Job {index}", "description": "legacy", "skills_required": []}
            for index in range(3)
        ]
        hybrid_jobs = [
            {
                "id": "api-1",
                "title": "Fresh Data Scientist",
                "description": "Python ML role",
                "skills_required": ["Python"],
                "source": "adzuna",
            }
        ]
        source_counts = {"db": 0, "external": 1, "db_ms": 1, "external_ms": 10}

        with patch.object(matching, "get_or_create_profile_embedding", return_value=[1.0, 0.0]), \
             patch.object(matching, "query_ann_jobs", return_value=ann_jobs), \
             patch.object(matching, "fetch_jobs_hybrid_async", new=AsyncMock(return_value=(hybrid_jobs, source_counts))) as fetch_mock, \
             patch.object(matching, "ensure_job_metadata", side_effect=lambda job, allow_llm=False: job), \
             patch.object(matching, "keyword_filter_jobs", side_effect=lambda profile, jobs, limit=40: jobs[:limit]), \
             patch.object(
                 matching,
                 "semantic_rank_jobs",
                 return_value=(
                     hybrid_jobs + ann_jobs,
                     {
                         "job_vectors_from_store": 0,
                         "job_vectors_embedded": 0,
                         "job_vectors_deferred": 0,
                         "embedding_ms": 0,
                         "similarity_ms": 0,
                     },
                 ),
             ):
            jobs, breakdown = await matching.retrieve_candidates(
                profile=profile,
                db=object(),
                jobs=None,
                fetch_limit=5,
                keyword_limit=3,
                similarity_limit=3,
            )

        self.assertEqual(fetch_mock.await_count, 1)
        self.assertIn("api-1", {job["id"] for job in jobs})
        self.assertEqual(breakdown["stages"]["source_external"], 1)


if __name__ == "__main__":
    unittest.main()
