import os
import requests
import json
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
logger = logging.getLogger(__name__)

def fetch_linkedin_serper_jobs(query: str, country: str = "fr", location: str = None, limit: int = 10, date_posted: str = "past_week") -> List[Dict[str, Any]]:
    """
    Search for LinkedIn jobs via Google using Serper API.
    """
    if not SERPER_API_KEY:
        logger.error("Missing SERPER_API_KEY. Set it in .env.")
        return []

    search_query = f'site:linkedin.com/jobs/view "{query}"'
    if location:
        search_query += f' "{location}"'

    url = "https://google.serper.dev/search"
    
    tbs_mapping = {
        "past_24h": "qdr:d",
        "past_week": "qdr:w",
        "past_month": "qdr:m",
        "any": None
    }
    
    payload_dict = {
        "q": search_query,
        "num": limit,
        "gl": country
    }
    
    tbs_val = tbs_mapping.get(date_posted)
    if tbs_val:
        payload_dict["tbs"] = tbs_val
        
    payload = json.dumps(payload_dict)
    
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()
        organic = results.get("organic", [])
        
        jobs = []
        for result in organic:
            link = result.get("link")
            if not link or "linkedin.com/jobs/view" not in link:
                continue
            
            logger.info(f"Crawling LinkedIn job: {link}")
            scraped_data = scrape_linkedin_job(link)
            
            if scraped_data:
                jobs.append(scraped_data)
            else:
                # Fallback to search result snippet if scraping fails (e.g. anti-bot)
                job_id = link.split("/")[-1].split("?")[0]
                jobs.append({
                    "id": job_id,
                    "title": result.get("title", "Unknown Title"),
                    "company": "LinkedIn", # Placeholder if not in snippet
                    "location": location or "Remote",
                    "description": result.get("snippet", ""),
                    "url": link,
                    "source": "linkedin_serper_snippet"
                })
        
        return jobs
    except Exception as e:
        logger.error(f"Serper API request failed: {e}")
        return []

def scrape_linkedin_job(url: str) -> Dict[str, Any]:
    """
    Scrapes a LinkedIn job page to extract structured data via JSON-LD or HTML.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch LinkedIn page {url}: Status {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Try JSON-LD (Standard for Job Posting)
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    # Sometimes LD+JSON is a list
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') in ('JobPosting', 'http://schema.org/JobPosting'):
                            return _parse_json_ld(item, url)
                elif isinstance(data, dict) and data.get('@type') in ('JobPosting', 'http://schema.org/JobPosting'):
                    return _parse_json_ld(data, url)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        
        # 2. Fallback: CSS Selectors (Modern LinkedIn Public Page)
        return _parse_html_fallback(soup, url)
        
    except Exception as e:
        logger.error(f"Error scraping LinkedIn job {url}: {e}")
        return None

def _parse_json_ld(data: Dict[str, Any], url: str) -> Dict[str, Any]:
    # Extract ID from URL
    job_id = url.split("/")[-1].split("?")[0]
    
    # Handle nested organization
    org = data.get("hiringOrganization")
    company = "Unknown Company"
    if isinstance(org, dict):
        company = org.get("name", "Unknown Company")
    elif isinstance(org, str):
        company = org

    # Handle location
    loc = data.get("jobLocation")
    location = "Remote"
    if isinstance(loc, dict):
        address = loc.get("address", {})
        if isinstance(address, dict):
            location = address.get("addressLocality") or address.get("addressRegion") or "Global"
        
    return {
        "id": job_id,
        "title": data.get("title"),
        "company": company,
        "location": location,
        "description": data.get("description"),
        "posted_at": data.get("datePosted"),
        "url": url,
        "source": "linkedin_serper_ldjson"
    }

def _parse_html_fallback(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    # Extract ID from URL
    job_id = url.split("/")[-1].split("?")[0]

    # Try diverse selectors common in LinkedIn public pages
    title_tag = soup.find('h1') or soup.find('h2', class_='topcard__title')
    title = title_tag.text.strip() if title_tag else "Unknown Title"
    
    company_tag = (
        soup.find('a', class_='topcard__org-name-link') or 
        soup.find('span', class_='topcard__flavor') or 
        soup.find('div', class_='top-card-layout__card')
    )
    company = company_tag.text.strip() if company_tag else "Unknown Company"
    
    # Description parsing
    desc_tag = (
        soup.find('div', class_='description__text') or 
        soup.find('div', class_='show-more-less-html__markup') or
        soup.find('section', class_='description')
    )
    description = desc_tag.get_text(separator="\n").strip() if desc_tag else ""
    
    return {
        "id": job_id,
        "title": title,
        "company": company,
        "location": "Remote / Check URL",
        "description": description,
        "url": url,
        "source": "linkedin_serper_html"
    }
