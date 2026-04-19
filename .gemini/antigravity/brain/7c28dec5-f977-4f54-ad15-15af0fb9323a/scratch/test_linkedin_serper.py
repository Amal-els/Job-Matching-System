import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "ai-service"))

from ingestion.linkedin_serper import fetch_linkedin_serper_jobs, scrape_linkedin_job
from services.matching import fetch_jobs_hybrid_async

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_scraper():
    # Test a public job URL (sample)
    test_url = "https://www.linkedin.com/jobs/view/3872421685/" # Replace with a recent one if this fails
    logger.info(f"Testing scraper with URL: {test_url}")
    result = scrape_linkedin_job(test_url)
    if result:
        logger.info(f"Scrape Success! Title: {result.get('title')}, Company: {result.get('company')}")
        logger.info(f"Description length: {len(result.get('description', ''))}")
    else:
        logger.error("Scraper failed (might be blocked or invalid URL)")

async def test_serper_search():
    query = "Software Engineer"
    location = "Paris"
    logger.info(f"Testing Serper search for '{query}' in '{location}'")
    jobs = fetch_linkedin_serper_jobs(query, location=location, limit=3)
    logger.info(f"Found {len(jobs)} jobs via Serper")
    for job in jobs:
        logger.info(f"- [{job.get('source')}] {job.get('title')} at {job.get('company')} ({job.get('url')})")

async def test_hybrid_fetch():
    from database import SessionLocal
    db = SessionLocal()
    try:
        query = "Python Developer"
        logger.info(f"Testing Hybrid Fetch (DB + Adzuna + LinkedIn) for '{query}'")
        jobs, stats = await fetch_jobs_hybrid_async(db, query, limit=10)
        logger.info(f"Hybrid Fetch Stats: {stats}")
        logger.info(f"Total jobs returned: {len(jobs)}")
    finally:
        db.close()

if __name__ == "__main__":
    # Run tests
    # asyncio.run(test_scraper())
    asyncio.run(test_serper_search())
    # asyncio.run(test_hybrid_fetch())
