# app/routes/__init__.py
from .csv_route import main as csv_blueprint
from .gat_route import gat as gat_blueprint
from .neo4j_route import graph as neo4j_blueprint

def register_routes(app):
    app.register_blueprint(csv_blueprint)
    app.register_blueprint(gat_blueprint)
    app.register_blueprint(neo4j_blueprint)