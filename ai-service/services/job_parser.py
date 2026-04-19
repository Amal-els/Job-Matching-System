import os
import re

from services.llm_client import chat_json, get_llm_client, is_enabled


SKILL_PATTERNS = [
    ("Python", r"\bpython\b"),
    ("Java", r"\bjava\b"),
    ("JavaScript", r"\bjavascript\b"),
    ("TypeScript", r"\btypescript\b"),
    ("C++", r"\bc\+\+\b"),
    ("C#", r"\bc#\b|\bcsharp\b"),
    ("SQL", r"\bsql\b"),
    ("PostgreSQL", r"\bpostgres(?:ql)?\b"),
    ("MySQL", r"\bmysql\b"),
    ("MongoDB", r"\bmongodb\b"),
    ("Redis", r"\bredis\b"),
    ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\bkubernetes\b|\bk8s\b"),
    ("AWS", r"\baws\b|\bamazon web services\b"),
    ("Azure", r"\bazure\b"),
    ("GCP", r"\bgcp\b|\bgoogle cloud\b"),
    ("Git", r"\bgit\b"),
    ("Linux", r"\blinux\b"),
    ("FastAPI", r"\bfastapi\b"),
    ("Django", r"\bdjango\b"),
    ("Flask", r"\bflask\b"),
    ("React", r"\breact(?:\.js)?\b"),
    ("Next.js", r"\bnext\.?js\b"),
    ("Node.js", r"\bnode\.?js\b"),
    ("Express", r"\bexpress\b"),
    ("HTML", r"\bhtml\b"),
    ("CSS", r"\bcss\b"),
    ("Tailwind", r"\btailwind\b"),
    ("REST API", r"\brest(?:ful)? api\b|\brest\b"),
    ("GraphQL", r"\bgraphql\b"),
    ("Pandas", r"\bpandas\b"),
    ("NumPy", r"\bnumpy\b"),
    ("TensorFlow", r"\btensorflow\b"),
    ("PyTorch", r"\bpytorch\b"),
    ("Machine Learning", r"\bmachine learning\b"),
]

SENIORITY_PATTERNS = [
    ("intern", [r"\bintern(ship)?\b", r"\btrainee\b", r"\bapprentice\b"]),
    ("junior", [r"\bjunior\b", r"\bentry[- ]level\b", r"\bgraduate\b", r"\b0[- ]?2 years?\b"]),
    ("mid", [r"\bmid[- ]level\b", r"\bintermediate\b", r"\b2[- ]?5 years?\b"]),
    ("senior", [r"\bsenior\b", r"\bsr\.?\b", r"\b5\+ years?\b", r"\bconfirmed\b"]),
    ("lead", [r"\blead\b", r"\bprincipal\b", r"\bstaff\b", r"\barchitect\b", r"\bmanager\b"]),
]

INDUSTRY_PATTERNS = [
    ("AI / Data", [r"\bartificial intelligence\b", r"\bmachine learning\b", r"\bdata science\b", r"\bcomputer vision\b"]),
    ("Finance", [r"\bfintech\b", r"\bbanking\b", r"\bpayments?\b", r"\binsurance\b"]),
    ("Healthcare", [r"\bhealthcare\b", r"\bmedical\b", r"\bpharma\b", r"\bclinical\b"]),
    ("E-commerce", [r"\be-?commerce\b", r"\bonline retail\b", r"\bmarketplace\b"]),
    ("Education", [r"\bedtech\b", r"\beducation\b", r"\blearning\b"]),
    ("Cybersecurity", [r"\bcybersecurity\b", r"\bsecurity operations\b", r"\biam\b"]),
    ("Telecom", [r"\btelecom\b", r"\btelecommunications\b", r"\bnetwork operator\b"]),
    ("Consulting", [r"\bconsulting\b", r"\bconsultant\b", r"\bclient projects\b"]),
    ("SaaS", [r"\bsaas\b", r"\bsoftware platform\b", r"\bb2b software\b"]),
]

VALID_SENIORITY_LEVELS = {"intern", "junior", "mid", "senior", "lead"}


def normalize_text(*parts):
    return "\n".join(str(part).strip() for part in parts if part).strip()


def _unique_strings(values):
    unique = []
    for value in values or []:
        cleaned = str(value).strip()
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def extract_skills_heuristic(text):
    normalized = (text or "").lower()
    skills = []

    for skill, pattern in SKILL_PATTERNS:
        if re.search(pattern, normalized) and skill not in skills:
            skills.append(skill)

    return skills


def extract_seniority_heuristic(text):
    normalized = (text or "").lower()

    for label, patterns in SENIORITY_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label

    return None


def extract_industry_heuristic(text):
    normalized = (text or "").lower()

    for label, patterns in INDUSTRY_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label

    return None


def _normalize_llm_metadata(payload):
    skills = payload.get("skills_required") or payload.get("skills") or []
    seniority = payload.get("seniority")
    industry = payload.get("industry")

    if isinstance(skills, list):
        skills = _unique_strings(skills)
    else:
        skills = _unique_strings([skills])

    seniority = str(seniority).strip().lower() if seniority else None
    if seniority not in VALID_SENIORITY_LEVELS:
        seniority = None

    industry = str(industry).strip() if industry else None

    return {
        "skills_required": skills,
        "seniority": seniority,
        "industry": industry,
    }


def extract_job_metadata_llm(title, description):
    model_name = os.getenv("JOB_METADATA_MODEL_NAME") or os.getenv("RERANKER_MODEL_NAME") or "openai/gpt-oss-120b:free"
    prompt = f"""
Extract structured hiring metadata from this job offer.

Return only valid JSON with this exact shape:
{{
  "skills_required": ["skill 1", "skill 2"],
  "seniority": "intern | junior | mid | senior | lead | null",
  "industry": "industry label or null"
}}

Rules:
- `skills_required` should contain only concrete technical or role-relevant skills.
- `seniority` must be one of: intern, junior, mid, senior, lead, or null.
- `industry` should be a short label like "Finance", "Healthcare", "SaaS", "E-commerce", "AI / Data", or null.
- Do not include explanations or markdown.

Job title:
{title or ""}

Job description:
{description or ""}
"""

    payload = chat_json(
        model=model_name,
        system_prompt="You extract structured metadata from job offers and return strict JSON only.",
        user_prompt=prompt,
        temperature=0,
    )
    return _normalize_llm_metadata(payload)


def extract_job_metadata(title, description, allow_llm=True):
    text = normalize_text(title or "", description or "")
    heuristic = {
        "skills_required": extract_skills_heuristic(text),
        "seniority": extract_seniority_heuristic(text),
        "industry": extract_industry_heuristic(text),
    }

    should_use_llm = allow_llm and is_enabled("USE_LLM_JOB_METADATA", "true")
    if should_use_llm and get_llm_client():
        try:
            llm_metadata = extract_job_metadata_llm(title, description)
            return {
                "skills_required": llm_metadata["skills_required"] or heuristic["skills_required"],
                "seniority": llm_metadata["seniority"] or heuristic["seniority"],
                "industry": llm_metadata["industry"] or heuristic["industry"],
            }
        except Exception:
            pass

    return heuristic
