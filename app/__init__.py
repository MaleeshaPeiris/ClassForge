# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import DevelopmentConfig, ProductionConfig, TestingConfig
import os
from dotenv import load_dotenv
from flask_migrate import Migrate

load_dotenv()

db = SQLAlchemy()


# after db = SQLAlchemy()
migrate = Migrate()

# Initialize the database and migration engine
def create_app():
    app = Flask(__name__)

    app.secret_key = os.getenv("FLASK_SECRET_KEY", "classforge-secret-key")


    # Select config based on environment
    env = os.getenv("FLASK_ENV", "development")
    if env == "production":
        app.config.from_object(ProductionConfig)
    elif env == "testing":
        app.config.from_object(TestingConfig)
    else:
        app.config.from_object(DevelopmentConfig)

    # Initialize database
    db.init_app(app)
    migrate.init_app(app, db)   # ✅ Enable migrations

    # Register all routes in one place
    from app.routes import register_routes
    register_routes(app)

    return app
