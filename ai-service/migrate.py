from sqlalchemy import text
from database import engine

def migrate():
    print("Starting migration...")
    with engine.connect() as conn:
        # Add missing columns to job_offer if they don't exist
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS posted_at VARCHAR;"))
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS url VARCHAR;"))
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;"))
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS seniority VARCHAR;"))
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS industry VARCHAR;"))
        conn.execute(text("ALTER TABLE job_offer ADD COLUMN IF NOT EXISTS vector JSONB;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id VARCHAR PRIMARY KEY,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                vector DOUBLE PRECISION[],
                profile_text_hash VARCHAR NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
