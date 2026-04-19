from services.job_parser import extract_job_metadata


def map_adzuna_job(job, allow_llm=True):
    title = job.get("title") or ""
    description = job.get("description") or ""
    location = job.get("location", {}).get("display_name") or ""
    metadata = extract_job_metadata(title, description, allow_llm=allow_llm)

    return {
        "id": f"adzuna_{job['id']}",
        "title": title,
        "company": job.get("company", {}).get("display_name"),
        "location": location,
        "remote": "remote" in location.lower(),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "contract_type": job.get("contract_type"),
        "description": description,
        "skills_required": metadata["skills_required"],
        "seniority": metadata["seniority"],
        "industry": metadata["industry"],
        "source": "adzuna",
        "posted_at": job.get("created"),
        "url": job.get("redirect_url"),
        "vector": None,
    }
