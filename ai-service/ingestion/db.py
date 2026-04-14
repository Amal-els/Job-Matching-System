import os
from pathlib import Path

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

database_url = os.getenv("INGESTION_DATABASE_URL") or os.getenv("DATABASE_URL")
if not database_url:
    raise RuntimeError("Missing database configuration. Set INGESTION_DATABASE_URL or DATABASE_URL.")

conn = psycopg2.connect(database_url)

conn.autocommit = True
cur = conn.cursor()
def insert_job(job):
    query = """
            INSERT INTO job_offer (
                id, title, company, location, remote,
                salary_min, salary_max, contract_type,
                description, skills_required,
                source, posted_at, url, vector
            )
            VALUES (
                %(id)s, %(title)s, %(company)s, %(location)s, %(remote)s,
                %(salary_min)s, %(salary_max)s, %(contract_type)s,
                %(description)s, %(skills_required)s,
                %(source)s, %(posted_at)s, %(url)s, %(vector)s
            )
            ON CONFLICT (id) DO NOTHING;
            """

    cur.execute(query, {
        **job,
        "skills_required": Json(job.get("skills_required", []))
    })
