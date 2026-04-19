from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from deps import get_db
from services.profile_store import get_profile_record, save_profile, serialize_profile_record


router = APIRouter()


def _extract_profile(payload, fallback_id=None):
    profile = payload.get("user") or payload.get("profile") or payload
    if not isinstance(profile, dict):
        raise ValueError("Provide a `user` or `profile` object.")

    extracted = dict(profile)
    if fallback_id and not extracted.get("id"):
        extracted["id"] = str(fallback_id)
    return extracted


@router.post("/")
def create_or_update_profile(payload: dict, db: Session = Depends(get_db)):
    try:
        profile = _extract_profile(payload)
        stored_profile, created = save_profile(db, profile)
        stored_profile["created"] = created
        return stored_profile
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/{profile_id}")
def update_profile(profile_id: str, payload: dict, db: Session = Depends(get_db)):
    try:
        profile = _extract_profile(payload, fallback_id=profile_id)
        stored_profile, _ = save_profile(db, profile)
        stored_profile["created"] = False
        return stored_profile
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{profile_id}")
def get_profile(profile_id: str, db: Session = Depends(get_db)):
    record = get_profile_record(db, profile_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return serialize_profile_record(record)
