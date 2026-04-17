from sqlalchemy import text
from database import engine

def migrate():
    print("Starting migration...")
    with engine.connect() as conn:
        # Add missing columns if they don't exist
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS posted_at VARCHAR;"))
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS url VARCHAR;"))
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP;"))
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS embedding JSONB;"))
        conn.commit()
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
