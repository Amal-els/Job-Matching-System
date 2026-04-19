from sqlalchemy import Column, String, DateTime, JSON, ARRAY, Float
from sqlalchemy.sql import func

from database import Base


class Profile(Base):
    __tablename__ = "user_profile"

    id = Column(String, primary_key=True)
    payload = Column(JSON, nullable=False)
    vector = Column(ARRAY(Float), nullable=True)
    profile_text_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
