# gat_pipeline.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from neo4j import GraphDatabase
import json
import os

# -------------------------
# GAT MODEL DEFINITIONS
# -------------------------
class DualHeadGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_node_dim, out_edge_dim, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)

        self.node_predictor = torch.nn.Linear(hidden_channels * heads, out_node_dim)  # exam score
        self.edge_predictor = torch.nn.Bilinear(hidden_channels * heads, hidden_channels * heads, out_edge_dim)  # edge type

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        node_preds = self.node_predictor(x)  # shape: [num_nodes, 1]

        edge_preds = []
        for src, tgt in edge_index.t():
            edge_pred = self.edge_predictor(x[src], x[tgt])  # shape: [out_edge_dim]
            edge_preds.append(edge_pred)
        edge_preds = torch.stack(edge_preds)

        return node_preds, edge_preds, x  # x is embedding


# -------------------------
# EDGE TYPE MAPPER
# -------------------------
EDGE_LABELS = ["Friend", "Neutral", "Conflict"]

def classify_edge_type(pred_tensor):
    return EDGE_LABELS[pred_tensor.argmax().item()]


# -------------------------
# GAT RUNNER + EXPORT
# -------------------------
def run_gat_and_export(data, neo4j_config, d3_path):
    print("🔄 Initializing DualHeadGAT model...")
    model = DualHeadGAT(
        in_channels=data.x.shape[1],
        hidden_channels=32,
        out_node_dim=1,
        out_edge_dim=3
    )

    model.eval()
    with torch.no_grad():
        print("🔍 Running forward pass...")
        node_preds, edge_preds, embeddings = model(data.x, data.edge_index)

    print("📊 Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings.numpy())
    cluster_labels = kmeans.labels_

    print("📦 Formatting node and edge data...")
    nodes = []
    for idx, embedding in enumerate(embeddings):
        nodes.append({
            "id": str(idx),
            "cluster": int(cluster_labels[idx]),
            "score": float(node_preds[idx].item()),
            "embedding": embedding.tolist()
        })

    edges = []
    for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
        edge_type = classify_edge_type(edge_preds[i])
        edges.append({"source": str(src), "target": str(tgt), "type": edge_type})

    print("💾 Saving to D3 JSON file...")
    with open(d3_path, "w") as f:
        json.dump({"nodes": nodes, "links": edges}, f, indent=2)

    print("🌐 Connecting to Neo4j...")
    driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
    with driver.session() as session:
        print("🧹 Clearing existing graph...")
        session.run("MATCH (n) DETACH DELETE n")

        print("🔁 Uploading nodes to Neo4j...")
        for node in nodes:
            session.run("""
                CREATE (s:Student {id: $id, cluster: $cluster, score: $score, embedding: $embedding})
            """, node)

        print("🔁 Uploading edges to Neo4j...")
        for edge in edges:
            session.run("""
                MATCH (a:Student {id: $source}), (b:Student {id: $target})
                CREATE (a)-[:RELATES {type: $type}]->(b)
            """, edge)

    print("✅ GAT embeddings, clusters, and edges exported to Neo4j and JSON.")


# # gat_pipeline.py

# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from sklearn.cluster import KMeans
# from neo4j import GraphDatabase
# import json
# import os

# # -------------------------
# # GAT MODEL DEFINITIONS
# # -------------------------
# class DualHeadGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_node_dim, out_edge_dim, heads=1):
#         super().__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
#         self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)

#         self.node_predictor = torch.nn.Linear(hidden_channels * heads, out_node_dim)  # exam score
#         self.edge_predictor = torch.nn.Bilinear(hidden_channels * heads, hidden_channels * heads, out_edge_dim)  # edge type

#     def forward(self, x, edge_index):
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.elu(self.gat2(x, edge_index))

#         node_preds = self.node_predictor(x)  # shape: [num_nodes, 1]

#         edge_preds = []
#         for src, tgt in edge_index.t():
#             edge_pred = self.edge_predictor(x[src], x[tgt])  # shape: [out_edge_dim]
#             edge_preds.append(edge_pred)
#         edge_preds = torch.stack(edge_preds)

#         return node_preds, edge_preds, x  # x is embedding


# # -------------------------
# # EDGE TYPE MAPPER
# # -------------------------
# EDGE_LABELS = ["Friend", "Neutral", "Conflict"]

# def classify_edge_type(pred_tensor):
#     return EDGE_LABELS[pred_tensor.argmax().item()]


# # -------------------------
# # GAT RUNNER + EXPORT
# # -------------------------
# def run_gat_and_export(data, neo4j_config, d3_path):
#     model = DualHeadGAT(
#         in_channels=data.x.shape[1],
#         hidden_channels=32,
#         out_node_dim=1,
#         out_edge_dim=3
#     )

#     model.eval()
#     with torch.no_grad():
#         node_preds, edge_preds, embeddings = model(data.x, data.edge_index)

#     # Cluster embeddings for classroom grouping
#     kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings.numpy())
#     cluster_labels = kmeans.labels_

#     # Format for Neo4j and D3
#     nodes = []
#     for idx, embedding in enumerate(embeddings):
#         nodes.append({
#             "id": str(idx),
#             "cluster": int(cluster_labels[idx]),
#             "score": float(node_preds[idx].item()),
#             "embedding": embedding.tolist()
#         })

#     edges = []
#     for i, (src, tgt) in enumerate(data.edge_index.t().tolist()):
#         edge_type = classify_edge_type(edge_preds[i])
#         edges.append({"source": str(src), "target": str(tgt), "type": edge_type})

#     # Export to D3 JSON
#     with open(d3_path, "w") as f:
#         json.dump({"nodes": nodes, "links": edges}, f, indent=2)

#     # Export to Neo4j
#     driver = GraphDatabase.driver(neo4j_config["uri"], auth=(neo4j_config["user"], neo4j_config["password"]))
#     with driver.session() as session:
#         session.run("MATCH (n) DETACH DELETE n")
#         for node in nodes:
#             session.run("""
#                 CREATE (s:Student {id: $id, cluster: $cluster, score: $score, embedding: $embedding})
#             """, node)

#         for edge in edges:
#             session.run("""
#                 MATCH (a:Student {id: $source}), (b:Student {id: $target})
#                 CREATE (a)-[:RELATES {type: $type}]->(b)
#             """, edge)

#     print("✅ GAT embeddings, clusters, and edges exported to Neo4j and JSON.")
