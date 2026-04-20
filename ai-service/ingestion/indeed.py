import os
import requests
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
# Using '~' instead of '/' for the Apify API actor path
ACTOR_ID = "valig~indeed-jobs-scraper"
logger = logging.getLogger(__name__)

def fetch_indeed_jobs(query: str, location: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search for Indeed jobs via Apify using the valig/indeed-jobs-scraper actor.
    """
    if not APIFY_TOKEN:
        logger.error("Missing APIFY_TOKEN. Set it in .env.")
        return []

    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"

    # Specific input parameters depend on the actor. 'search' and 'location' are standard
    # valig/indeed-jobs-scraper usually expects 'position' and 'location'. Let's provide both just in case.
    payload = {
        "position": query,
        "search": query,
        "location": location or "Remote",
        "maxItems": limit
    }

    try:
        response = requests.post(
            url,
            params={"token": APIFY_TOKEN},
            json=payload,
            timeout=45 # Apify sync runs can take time
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Apify Indeed request failed: {e}")
        return []
