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


def map_serper_job(job, allow_llm=True):
    title = job.get("title") or ""
    description = job.get("description") or ""
    location = job.get("location") or "Remote"
    metadata = extract_job_metadata(title, description, allow_llm=allow_llm)

    return {
        "id": f"linkedin_{job['id']}",
        "title": title,
        "company": job.get("company", "Unknown Company"),
        "location": location,
        "remote": "remote" in location.lower() or "remote" in title.lower(),
        "salary_min": None,
        "salary_max": None,
        "contract_type": None,
        "description": description,
        "skills_required": metadata["skills_required"],
        "seniority": metadata["seniority"],
        "industry": metadata["industry"],
        "source": job.get("source", "linkedin_serper"),
        "posted_at": job.get("posted_at"),
        "url": job.get("url"),
        "vector": None,
    }


def map_indeed_job(job, allow_llm=True):
    title = job.get("positionName") or job.get("title") or ""
    if isinstance(title, dict):
        title = title.get("text", str(title))
    elif not isinstance(title, str):
        title = str(title)

    description = job.get("description") or ""
    if isinstance(description, dict):
        description = description.get("text", str(description))
    elif not isinstance(description, str):
        description = str(description)
    
    # Valig actor returns location in 'location', company in 'company'
    location = job.get("location") or "Remote"
    company = job.get("company") or "Unknown Company"
    if isinstance(company, dict):
        company = company.get("name", str(company))
    if isinstance(location, dict):
        location = location.get("text", str(location))
    
    metadata = extract_job_metadata(title, description, allow_llm=allow_llm)

    job_id = job.get("id") or job.get("jobkey") or str(abs(hash(title + company)))

    return {
        "id": f"indeed_{job_id}",
        "title": title,
        "company": company,
        "location": location,
        "remote": "remote" in location.lower() or "remote" in title.lower(),
        "salary_min": job.get("salaryMin") or job.get("salary_min"),
        "salary_max": job.get("salaryMax") or job.get("salary_max"),
        "contract_type": job.get("jobType") or job.get("contract_type"),
        "description": description,
        "skills_required": metadata["skills_required"],
        "seniority": metadata["seniority"],
        "industry": metadata["industry"],
        "source": "indeed",
        "posted_at": job.get("postedAt") or job.get("date"),
        "url": job.get("url"),
        "vector": None,
    }
