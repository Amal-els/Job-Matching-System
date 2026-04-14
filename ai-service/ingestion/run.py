import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ingestion.adzuna import fetch_adzuna_jobs
from ingestion.db import insert_job
from ingestion.mapper import map_adzuna_job


def run():
    query = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ADZUNA_QUERY", "software")
    country = os.getenv("ADZUNA_COUNTRY", "fr")
    jobs = fetch_adzuna_jobs(country=country, query=query, limit=50)

    for job in jobs:
        normalized = map_adzuna_job(job)
        insert_job(normalized)

    print(f"Inserted {len(jobs)} jobs for query '{query}'")


if __name__ == "__main__":
    run()
