import os
import requests
import json
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Add ai-service to path
import sys
sys.path.append(os.path.join(os.getcwd(), "ai-service"))

from ingestion.linkedin_serper import fetch_linkedin_serper_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test():
    query = "Software engineer"
    location = "Tunisia"
    logger.info(f"Testing Serper fetch for '{query}' in '{location}'")
    jobs = fetch_linkedin_serper_jobs(query, location=location, limit=2)
    print(f"Total jobs found: {len(jobs)}")
    for j in jobs:
        print(f"- {j.get('title')} @ {j.get('company')} | Source: {j.get('source')}")

if __name__ == "__main__":
    test()
