
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from routes.match import router as match_router
from routes.profile import router as profile_router
from database import Base, engine
from models.job import Job
from models.profile import Profile
from deps import get_db

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(match_router, prefix="/match")
app.include_router(profile_router, prefix="/profiles")

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    jobs = db.query(Job).all()
    return {"count": len(jobs)}


@app.post("/seed")
def seed(db: Session = Depends(get_db)):
    job = Job(
        id="1",
        title="Backend Developer",
        company="Google",
        location="Remote",
        remote=True,
        description="Build APIs",
        skills_required=["Python", "FastAPI"],
        source="manual"
    )

    db.add(job)
    db.commit()

    return {"message": "Inserted"}
