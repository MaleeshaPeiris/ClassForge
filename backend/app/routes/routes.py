import json
from flask import Blueprint, jsonify,request
from app.models import Student
from app.services.gat_model import run_gat_and_generate_graph
import os

main = Blueprint('main', __name__)

@main.route('/api/students')
def get_students():
    students = Student.query.all()
    return jsonify([s.serialize() for s in students])

# 🆕 NEW
@main.route('/api/network')
def get_network_graph():
    try:
        graph_data = run_gat_and_generate_graph()
        return jsonify(graph_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🆕 NEW: Serve classroom_graph_d3.json
@main.route('/api/graph')
def get_graph():
    json_path = os.path.join(os.path.dirname(__file__), '../classroom_graph_d3.json')
    with open(json_path, 'r') as f:
        graph_data = json.load(f)
    return jsonify(graph_data)