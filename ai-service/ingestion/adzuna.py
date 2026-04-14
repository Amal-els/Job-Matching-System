import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")


def fetch_adzuna_jobs(country="fr", page=1, query="software", limit=20, location=None):
    if not APP_ID or not APP_KEY:
        raise RuntimeError("Missing Adzuna credentials. Set ADZUNA_APP_ID and ADZUNA_APP_KEY.")

    jobs = []
    current_page = page

    while len(jobs) < limit:
        remaining = limit - len(jobs)
        page_size = min(20, remaining)
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{current_page}"
        params = {
            "app_id": APP_ID,
            "app_key": APP_KEY,
            "results_per_page": page_size,
            "what": query,
        }

        if location:
            params["where"] = location

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Adzuna API error: {response.text}")

        page_results = response.json().get("results", [])
        if not page_results:
            break

        jobs.extend(page_results)

        if len(page_results) < page_size:
            break

        current_page += 1

    return jobs[:limit]
