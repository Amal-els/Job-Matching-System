from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from deps import get_db

from services.matching import run_matching_pipeline
from services.profile_store import get_profile_payload

router = APIRouter()


@router.post("/")
async def match_jobs(payload: dict, db: Session = Depends(get_db)):
    profile = payload.get("user") or payload.get("profile")
    profile_id = payload.get("profile_id")
    if profile is None and profile_id:
        profile = get_profile_payload(db, profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Profile not found.")
    if not profile:
        raise HTTPException(status_code=400, detail="Provide a `user`, `profile`, or `profile_id`.")

    if not profile.get("target_position"):
        raise HTTPException(status_code=400, detail="`target_position` is required in the user profile.")

    try:
        return await run_matching_pipeline(
            profile=profile,
            db=db,
            jobs=payload.get("jobs"),
            country=payload.get("country", "fr"),
            fetch_limit=int(payload.get("fetch_limit", 50)),
            keyword_limit=int(payload.get("keyword_limit", 40)),
            similarity_limit=int(payload.get("similarity_limit", 25)),
            final_limit=int(payload.get("final_limit", 10)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
