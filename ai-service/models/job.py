from sqlalchemy import Column, String, Integer, Boolean, Text, JSON, DateTime
from sqlalchemy.sql import func
from database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    title = Column(String)
    company = Column(String)
    location = Column(String)
    remote = Column(Boolean)
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    contract_type = Column(String)
    description = Column(Text)
    skills_required = Column(JSON)
    source = Column(String)
    posted_at = Column(String)
    url = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
