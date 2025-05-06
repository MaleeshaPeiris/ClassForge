# app/services/gat_model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
import pandas as pd
import os

# Define Upgraded GAT model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x, (edge_index, alpha1) = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, (edge_index, alpha2) = self.gat2(x, edge_index, return_attention_weights=True)
        return x, alpha2  # Return output features and last attention

def run_gat_and_generate_graph():
    # Step 1: Load your real dataset
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, 'students_data.csv')  # 📝 Adjust this to your real CSV name
    df = pd.read_csv(csv_path)

    # Step 2: Select features
    features_df = df[['Motivation', 'Attendance']]  # Adjust these column names to match your dataset
    features = torch.tensor(features_df.values, dtype=torch.float)

    num_students = features.shape[0]

    # Step 3: Generate edges (you can replace this with smarter connection logic)
    edges = []
    for i in range(num_students):
        for j in range(i+1, num_students):
            if torch.rand(1) < 0.05:  # 5% random connection probability
                edges.append([i, j])
                edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Step 4: Instantiate and run GAT
    model = GAT(input_dim=features.shape[1], hidden_dim=8, output_dim=4)
    output, alpha = model(features, edge_index)

    # Save embeddings (optional)
    torch.save(output, os.path.join(current_dir, 'gat_embeddings.pt'))  # Save as torch file

    # Step 5: Build NetworkX graph
    G = nx.Graph()

    for i in range(num_students):
        G.add_node(i,
            motivation=float(features[i][0].item()),
            attendance=float(features[i][1].item()),
            cluster=int(torch.argmax(output[i]).item())
        )

    # Step 6: Create edges with attention weights
    alpha = alpha.squeeze().tolist()  # Convert attention tensor to list
    for idx, (src, tgt) in enumerate(edge_index.t().tolist()):
        G.add_edge(src, tgt, attention=float(alpha[idx]))

    # Step 7: Format nodes/links for D3.js
    nodes = []
    links = []
    for node_id, attr in G.nodes(data=True):
        nodes.append({
            "id": str(node_id),
            "name": f"Student {node_id}",
            "group": attr['cluster'],
            "motivation": round(attr['motivation'], 2),
            "attendance": round(attr['attendance'], 2)
        })

    for src, tgt, attr in G.edges(data=True):
        links.append({
            "source": str(src),
            "target": str(tgt),
            "strength": attr.get('attention', 0.5)  # attention score as strength
        })

    return {
        "nodes": nodes,
        "links": links
    }
