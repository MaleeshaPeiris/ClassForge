# app/routes/neo4j_route.py

from flask import Blueprint, jsonify
from neo4j import GraphDatabase
import os

graph = Blueprint("neo4j", __name__)

# Load Neo4j connection settings from environment or fallback
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpassword")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


@graph.route("/neo4j/status", methods=["GET"])
def check_neo4j():
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count = result.single()["count"]
            return jsonify({"status": "connected", "node_count": count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@graph.route("/neo4j/reset", methods=["POST"])
def reset_graph():
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        return jsonify({"message": "✅ Neo4j graph reset complete."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
