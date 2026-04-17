from database import SessionLocal
from models.job import Job
from services.embedding import embed_many
from services.job_text import build_job_text


BATCH_SIZE = 200


def run_backfill():
    db = SessionLocal()
    try:
        total_missing = db.query(Job).filter(Job.embedding.is_(None)).count()
        print(f"Jobs missing embedding: {total_missing}")
        if total_missing == 0:
            print("Nothing to backfill.")
            return

        updated = 0
        while True:
            rows = (
                db.query(Job)
                .filter(Job.embedding.is_(None))
                .order_by(Job.created_at.asc())
                .limit(BATCH_SIZE)
                .all()
            )
            if not rows:
                break

            payloads = []
            for row in rows:
                payloads.append(
                    {
                        "title": row.title,
                        "company": row.company,
                        "location": row.location,
                        "description": row.description,
                        "skills_required": row.skills_required,
                    }
                )

            vectors = embed_many([build_job_text(item) for item in payloads])
            for row, vector in zip(rows, vectors):
                row.embedding = vector

            db.commit()
            updated += len(rows)
            print(f"Backfilled {updated}/{total_missing}")

        print("Backfill complete.")
    finally:
        db.close()


if __name__ == "__main__":
    run_backfill()
