from flask import Blueprint, jsonify
from app.services.gat_pipeline import run_gat_and_export
from app.models.model import TransformedStudentData
from torch_geometric.data import Data
import torch
import os

gat = Blueprint("gat", __name__)

@gat.route("/run-gat", methods=["GET"])
def run_gat():
    try:
        # 1. Load student records from database
        students = TransformedStudentData.query.all()
        if not students:
            return jsonify({"error": "No transformed data found."}), 400

        # 2. Convert to feature matrix (x)
        features = []
        for s in students:
            features.append([
                s.encoded_gender,
                s.encoded_school_type,
                s.encoded_parental_education,
                s.normalized_score
            ])
        x = torch.tensor(features, dtype=torch.float)

        # 3. Create simple edge_index (connect every student to the next)
        src = list(range(len(students) - 1))
        tgt = list(range(1, len(students)))
        edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)  # bi-directional

        data = Data(x=x, edge_index=edge_index)

        # 4. Run export
        neo4j_config = {
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASS", "test")
        }

        d3_path = os.path.join("classroom_graph_d3.json")
        run_gat_and_export(data, neo4j_config, d3_path)

        return jsonify({"message": "✅ GAT ran and exported based on uploaded data."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
