# backend/app/models/student_clean.py

from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class StudentClean(Base):
    __tablename__ = 'students_clean'

    id = Column(Integer, primary_key=True, autoincrement=True)
    motivation_level = Column(Float)
    peer_influence_positive = Column(Float)
    peer_influence_negative = Column(Float)
    attendance = Column(Float)
    parental_involvement = Column(Float)
    sleep_hours = Column(Float)
    extracurricular_activities = Column(Float)
    tutoring_sessions = Column(Float)
    internet_access = Column(Float)
    school_type_private = Column(Integer)
    exam_score = Column(Float)  # Optional, can be used for evaluation

# Script to create the table in database
if __name__ == "__main__":
    from sqlalchemy import create_engine
    engine = create_engine("postgresql://postgres:database123@localhost:5432/classforge")
    Base.metadata.create_all(engine)
    print("✅ students_clean table created in PostgreSQL")
