from database import SessionLocal
from models.job import Job
from services.embedding import embed_many
from services.job_text import build_job_text
from services.matching import ensure_job_metadata


BATCH_SIZE = 10


def run_backfill():
    db = SessionLocal()
    try:
        total_missing = db.query(Job).filter(Job.vector.is_(None)).count()
        print(f"Jobs missing vector embedding: {total_missing}")
        if total_missing == 0:
            print("Nothing to backfill.")
            return

        updated = 0
        while True:
            rows = (
                db.query(Job)
                .filter(Job.vector.is_(None))
                .order_by(Job.id.asc())
                .limit(BATCH_SIZE)
                .all()
            )
            if not rows:
                break

            # 1. First, ensure all rows have metadata (seniority/industry)
            # This triggers LLM calls if needed
            for i, row in enumerate(rows):
                print(f"[{updated + i + 1}/{total_missing}] Processing metadata for {row.id}... ", end="", flush=True)
                if not row.seniority or not row.industry:
                    mapped_dict = {
                        "id": row.id,
                        "title": row.title,
                        "company": row.company,
                        "location": row.location,
                        "description": row.description,
                        "skills_required": row.skills_required,
                        "seniority": row.seniority,
                        "industry": row.industry,
                    }
                    updated_meta = ensure_job_metadata(mapped_dict)
                    row.seniority = updated_meta.get("seniority")
                    row.industry = updated_meta.get("industry")
                    row.skills_required = updated_meta.get("skills_required")
                print("Done.", flush=True)

            # 2. Build texts after metadata is populated for better embeddings
            texts = [
                build_job_text({
                    "title": row.title,
                    "company": row.company,
                    "location": row.location,
                    "description": row.description,
                    "skills_required": row.skills_required,
                    "seniority": row.seniority,
                    "industry": row.industry,
                }) 
                for row in rows
            ]

            vectors = embed_many(texts)
            for row, vector in zip(rows, vectors):
                # Convert list/numpy to regular list if needed
                # SQLAlchemy with ARRAY(Float) handles list to array conversion.
                row.vector = list(vector) if hasattr(vector, "tolist") else vector

            db.commit()
            updated += len(rows)
            print(f"Backfilled {updated}/{total_missing}")

        print("Backfill complete.")
    finally:
        db.close()


if __name__ == "__main__":
    run_backfill()
