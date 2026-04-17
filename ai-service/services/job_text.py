"""Shared profile/job text builders for embeddings and scoring (single source of truth)."""


def _scalar_to_text(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip()


def flatten_text(value):
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return _scalar_to_text(value)

    if isinstance(value, dict):
        preferred_keys = [
            "name", "title", "label", "value", "skill", "role", "position",
            "company", "school", "degree", "language", "certificate", "certification",
            "description", "summary",
        ]
        preferred_parts = [
            _scalar_to_text(value.get(key))
            for key in preferred_keys
            if value.get(key) not in (None, "", [], {})
        ]
        if preferred_parts:
            return " | ".join(part for part in preferred_parts if part)

        parts = []
        for key, item in value.items():
            flattened = flatten_text(item)
            if flattened:
                parts.append(f"{key}: {flattened}")
        return " | ".join(parts)

    if isinstance(value, (list, tuple, set)):
        return " ; ".join(part for part in (flatten_text(item) for item in value) if part)

    return str(value)


def normalize_list(values):
    if not values:
        return []

    if isinstance(values, (str, int, float, bool, dict)):
        values = [values]

    normalized = []
    for value in values:
        flattened = flatten_text(value).strip()
        if flattened:
            normalized.append(flattened)

    return normalized


def build_profile_text(profile):
    parts = [
        f"Profile ID: {profile.get('id', '')}",
        f"Target position: {profile.get('target_position', '')}",
        f"Bio: {profile.get('bio', '')}",
        f"User level: {profile.get('user_level', '')}",
        f"Profile score: {profile.get('profil_score', '')}",
        f"Skills: {', '.join(normalize_list(profile.get('skills')))}",
        f"Experiences: {flatten_text(profile.get('experiences'))}",
        f"Education: {flatten_text(profile.get('education'))}",
        f"Languages: {', '.join(normalize_list(profile.get('languages')))}",
        f"Certifications: {', '.join(normalize_list(profile.get('certifications')))}",
        f"Short term goals: {flatten_text(profile.get('short_term_goals'))}",
        f"Long term goals: {flatten_text(profile.get('long_term_goals'))}",
        f"Location: {profile.get('location', '')}",
    ]

    return "\n".join(part for part in parts if part.split(":", 1)[1].strip())


def build_job_text(job):
    parts = [
        f"Title: {job.get('title', '')}",
        f"Company: {job.get('company', '')}",
        f"Location: {job.get('location', '')}",
        f"Description: {job.get('description', '')}",
        f"Skills: {', '.join(normalize_list(job.get('skills_required')))}",
        f"Seniority: {job.get('seniority', '')}",
        f"Industry: {job.get('industry', '')}",
    ]

    return "\n".join(part for part in parts if part.split(":", 1)[1].strip())
