
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from routes.match import router
from database import Base, engine
from models.job import Job
from deps import get_db

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(router, prefix="/match")

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