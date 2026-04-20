import sys
import json
import logging
from pathlib import Path

# Add the parent directory to the Python path so it can find ai_service
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.linkedin_serper import fetch_linkedin_serper_jobs

# Set up logging so we can see the scraper's output
logging.basicConfig(level=logging.INFO)

def run_test():
    print("Starting LinkedIn scraper test...")
    jobs = fetch_linkedin_serper_jobs(
        query="software engineer", 
        location="Tunis",
        limit=10
    )

    print(f"\n--- Found {len(jobs)} jobs ---")
    print(json.dumps(jobs, indent=10, ensure_ascii=False))

if __name__ == "__main__":
    run_test()
