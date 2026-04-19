import os
import logging
from typing import Dict, Any, List
from services.llm_client import chat_json

logger = logging.getLogger(__name__)

def extract_job_metadata(title: str, description: str, allow_llm: bool = True) -> Dict[str, Any]:
    """
    Extract structured metadata (skills, seniority, industry) from job title and description.
    Uses LLM if allowed, otherwise falls back to a basic heuristic.
    """
    default_metadata = {
        "skills_required": [],
        "seniority": "mid",
        "industry": "technology"
    }

    if not allow_llm or not os.getenv("USE_LLM_JOB_METADATA", "true").lower() in ("true", "1", "yes"):
        return _extract_metadata_heuristic(title, description)

    system_prompt = (
        "You are an expert HR data scientist. Your task is to extract job metadata from a job title and description.\n"
        "Return a JSON object with exactly these keys:\n"
        "- 'skills_required': (list of strings) key technical and soft skills.\n"
        "- 'seniority': (string) one of ['intern', 'junior', 'mid', 'senior', 'lead'].\n"
        "- 'industry': (string) the primary industry sector (e.g., 'Finance', 'Healthcare', 'Software').\n"
        "Be concise and return ONLY the JSON object."
    )
    user_prompt = f"Job Title: {title}\nJob Description: {description[:1000]}"

    try:
        model = os.getenv("JOB_METADATA_MODEL_NAME", "llama-3.1-8b-instant")
        metadata = chat_json(model, system_prompt, user_prompt)
        
        # Validation and normalization
        if not isinstance(metadata, dict):
            raise ValueError("LLM returned non-dict metadata")
            
        # Ensure seniority is valid
        valid_seniorities = ["intern", "junior", "mid", "senior", "lead"]
        res_seniority = str(metadata.get("seniority", "mid")).lower()
        if res_seniority not in valid_seniorities:
            # Try to map if possible or default to mid
            for s in valid_seniorities:
                if s in res_seniority:
                    res_seniority = s
                    break
            else:
                res_seniority = "mid"
        
        return {
            "skills_required": metadata.get("skills_required", []),
            "seniority": res_seniority,
            "industry": metadata.get("industry", "Technology")
        }
    except Exception as e:
        logger.warning(f"LLM metadata extraction failed for '{title}': {e}. Falling back to heuristics.")
        return _extract_metadata_heuristic(title, description)

def _extract_metadata_heuristic(title: str, description: str) -> Dict[str, Any]:
    """Basic rule-based extraction fallback."""
    title_lower = title.lower()
    text = (title + " " + description).lower()
    
    seniority = "mid"
    if any(word in title_lower for word in ["intern", "trainee", "stagiere"]):
        seniority = "intern"
    elif any(word in title_lower for word in ["junior", "entry", "associate", "debutant", "graduate"]):
        seniority = "junior"
    elif any(word in title_lower for word in ["senior", "sr.", "confirmé", "expert", "sr"]):
        seniority = "senior"
    elif any(word in title_lower for word in ["lead", "staff", "principal", "manager", "director", "head", "chef", "cto", "architect"]):
        seniority = "lead"

    # Simple skill extraction (just a few examples for heuristic)
    common_skills = ["python", "java", "javascript", "react", "node", "aws", "docker", "kubernetes", "sql", "rust", "go", "c++", "c#", "typescript", "angular", "vue", "fastapi"]
    skills = [skill for skill in common_skills if skill in text]

    return {
        "skills_required": list(set(skills)),
        "seniority": seniority,
        "industry": "Technology"
    }