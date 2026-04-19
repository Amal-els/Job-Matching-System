import hashlib
import os
from datetime import datetime, timezone

from models.profile import Profile
from services.cache import get_cache
from services.embedding import embed
from services.job_text import build_profile_text


PROFILE_EMBED_TTL_SECONDS = int(os.getenv("PROFILE_EMBED_TTL_SECONDS", "3600"))


def _sanitize_profile_payload(profile):
    sanitized = dict(profile or {})
    sanitized.pop("vector", None)
    if sanitized.get("id") is not None:
        sanitized["id"] = str(sanitized["id"])
    return sanitized


def _profile_text_hash(profile_text):
    return hashlib.sha256((profile_text or "").encode("utf-8")).hexdigest()


def _profile_embedding_cache_key(profile_text):
    return f"profile_embedding:{_profile_text_hash(profile_text)}"


def _serialize_vector(vector):
    if vector is None:
        return None
    return list(vector) if hasattr(vector, "tolist") else vector


def serialize_profile_record(record, include_vector=True):
    payload = dict(record.payload or {})
    payload.setdefault("id", record.id)
    if include_vector and record.vector is not None:
        payload["vector"] = _serialize_vector(record.vector)
    return {
        "id": record.id,
        "profile": payload,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "embedding_precomputed": record.vector is not None,
    }


def get_profile_record(db, profile_id):
    if db is None or not profile_id:
        return None
    return db.query(Profile).filter(Profile.id == str(profile_id)).first()


def get_profile_payload(db, profile_id, include_vector=True):
    record = get_profile_record(db, profile_id)
    if record is None:
        return None
    payload = dict(record.payload or {})
    payload.setdefault("id", record.id)
    if include_vector and record.vector is not None:
        payload["vector"] = _serialize_vector(record.vector)
    return payload


def save_profile(db, profile):
    if db is None:
        raise ValueError("Database session is required.")

    sanitized = _sanitize_profile_payload(profile)
    profile_id = sanitized.get("id")
    if not profile_id:
        raise ValueError("`id` is required in the profile payload.")
    if not sanitized.get("target_position"):
        raise ValueError("`target_position` is required in the profile payload.")

    profile_text = build_profile_text(sanitized)
    profile_text_hash = _profile_text_hash(profile_text)
    existing = get_profile_record(db, profile_id)

    if existing is not None and existing.profile_text_hash == profile_text_hash and existing.vector is not None:
        vector = _serialize_vector(existing.vector)
    else:
        vector = embed(profile_text)

    created = existing is None
    if existing is None:
        record = Profile(
            id=profile_id,
            payload=sanitized,
            vector=vector,
            profile_text_hash=profile_text_hash,
        )
        db.add(record)
    else:
        record = existing
        record.payload = sanitized
        record.vector = vector
        record.profile_text_hash = profile_text_hash
        record.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(record)

    cache = get_cache()
    cache.set_json(_profile_embedding_cache_key(profile_text), vector, PROFILE_EMBED_TTL_SECONDS)

    return serialize_profile_record(record), created
