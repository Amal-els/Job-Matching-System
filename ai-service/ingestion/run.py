import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ingestion.adzuna import fetch_adzuna_jobs
from ingestion.linkedin_serper import fetch_linkedin_serper_jobs
from ingestion.db import insert_job
from ingestion.mapper import map_adzuna_job, map_serper_job


def run():
    query = sys.argv[1] if len(sys.argv) > 1 else os.getenv("ADZUNA_QUERY", "software")
    country = os.getenv("ADZUNA_COUNTRY", "fr")
    # 1. Fetch from Adzuna
    adzuna_jobs = fetch_adzuna_jobs(country=country, query=query, limit=50)
    for job in adzuna_jobs:
        normalized = map_adzuna_job(job)
        insert_job(normalized)
    print(f"Inserted {len(adzuna_jobs)} Adzuna jobs for query '{query}'")

    # 2. Fetch from LinkedIn via Serper
    linkedin_jobs = fetch_linkedin_serper_jobs(query=query, country=country, limit=10)
    for job in linkedin_jobs:
        normalized = map_serper_job(job)
        insert_job(normalized)
    print(f"Inserted {len(linkedin_jobs)} LinkedIn jobs for query '{query}'")


if __name__ == "__main__":
    run()
