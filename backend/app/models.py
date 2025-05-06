from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Student(Base):
    __tablename__ = 'students'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    motivation_level = Column(Float)
    peer_influence_positive = Column(Float)
    peer_influence_negative = Column(Float)
    attendance = Column(Float)
    parental_involvement = Column(Float)
    sleep_hours = Column(Float)
    extracurricular_activities = Column(Float)
    tutoring_sessions = Column(Float)
    internet_access = Column(Float)
    school_type_private = Column(Integer)  # 1 or 0
    exam_score = Column(Float)  # Optional for evaluation
