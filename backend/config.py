import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:database123@localhost:5432/classforge')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
