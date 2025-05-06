# backend/app/services/gat_allocation.py

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
import numpy as np


# -------------------------------
# ✅ Dual-Head GAT Architecture
# -------------------------------
class DualHeadGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super().__init__()  # Modern Python 3 style
        self.dropout = dropout

        # Shared GAT encoder (2 layers)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=True, dropout=dropout)

        # Head 1: Score regression (exam prediction)
        self.score_head = nn.Linear(hidden_channels, 1)

        # Head 2: Edge classification (friend/conflict/neutral)
        self.edge_head = nn.Linear(hidden_channels * 2, 3)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        scores = self.score_head(x).squeeze(-1)  # Ensure shape [N]
        return x, scores

    def classify_edges(self, node_embeddings, edge_index):
        # Get source and target node embeddings
        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]]

        # Concatenate for edge representation
        edge_repr = torch.cat([src, dst], dim=1)
        logits = self.edge_head(edge_repr)  # Shape: [E, 3]
        return logits


# -----------------------------------------
# ✅ Training Function for Dual-Head GAT
# -----------------------------------------
def train_dual_head_gat(data, labels, edge_index, edge_types=None, epochs=100, lr=0.001, supervised=False):
    # Convert inputs to torch tensors
    x = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Check supervised mode requirements
    if supervised and edge_types is None:
        raise ValueError("Supervised training requires edge type labels")

    edge_types_tensor = None
    if supervised:
        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)

    # Initialize model and optimizer
    model = DualHeadGAT(in_channels=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        node_embeddings, pred_scores = model(x, edge_index)

        # Loss 1: Score regression
        loss_score = F.mse_loss(pred_scores, y)

        # Loss 2: Edge type classification (if supervised)
        loss_edge = 0
        if supervised:
            edge_logits = model.classify_edges(node_embeddings, edge_index)
            loss_edge = F.cross_entropy(edge_logits, edge_types_tensor)

        # Combine losses
        loss = loss_score + (loss_edge if supervised else 0)
        loss.backward()
        optimizer.step()

        # Logging every 10 epochs
        if epoch % 10 == 0:
            log = f"Epoch {epoch} — Loss: {loss.item():.4f} | ScoreLoss: {loss_score.item():.4f}"
            if supervised:
                log += f" | EdgeLoss: {loss_edge.item():.4f}"
            print(log)

    return model


# -----------------------------------------
# ✅ Extract Embeddings from Trained Model
# -----------------------------------------
def extract_embeddings(model, data, edge_index):
    model.eval()
    x = torch.tensor(data, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    with torch.no_grad():
        node_embeddings, _ = model(x, edge_index)
    return node_embeddings.cpu().numpy()


# -----------------------------------------
# ✅ Cluster Embeddings into Groups (Classes)
# -----------------------------------------
def cluster_embeddings(embeddings, n_classes=10):
    kmeans = KMeans(n_clusters=n_classes, random_state=42)
    return kmeans.fit_predict(embeddings)






#Version 2
# # # backend/app/services/gat_allocation.py
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch_geometric.nn import GATConv
# from sklearn.cluster import KMeans
# import numpy as np


# # -------------------------------
# # ✅ Dual-Head GAT Architecture
# # -------------------------------
# class DualHeadGAT(nn.Module):
#     def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
#         # super(DualHeadGAT, self).__init__()
#         super().__init__()
#         self.dropout = dropout

#         # Shared encoder
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=4, dropout=dropout)
#         self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=True, dropout=dropout)

#         # Head 1: Score regression
#         self.score_head = nn.Linear(hidden_channels, 1)

#         # Head 2: Edge classification (friend/conflict/neutral → 3 classes)
#         self.edge_head = nn.Linear(hidden_channels * 2, 3)

#     def forward(self, x, edge_index):
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.elu(self.gat2(x, edge_index))
#         scores = self.score_head(x).squeeze()  # [N]
#         return x, scores

#     def classify_edges(self, node_embeddings, edge_index):
#         src = node_embeddings[edge_index[0]]
#         dst = node_embeddings[edge_index[1]]
#         edge_repr = torch.cat([src, dst], dim=1)
#         logits = self.edge_head(edge_repr)
#         return logits  # shape: [E, 3]
    

# # -----------------------------------------
# # ✅ Training Function for Dual-Head GAT
# # -----------------------------------------
# def train_dual_head_gat(data, labels, edge_index, edge_types=None, epochs=100, lr=0.001, supervised=False):
#     x = torch.tensor(data, dtype=torch.float)
#     y = torch.tensor(labels, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long)

#     if supervised and edge_types is None:
#         raise ValueError("Supervised training requires edge type labels")

#     edge_types_tensor = None
#     if supervised:
#         edge_types_tensor = torch.tensor(edge_types, dtype=torch.long)  # shape: [E]

#     model = DualHeadGAT(in_channels=x.shape[1])
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         node_embeddings, pred_scores = model(x, edge_index)

#         # Loss 1: Exam score prediction (MSE)
#         loss_score = F.mse_loss(pred_scores, y)

#         # Loss 2: Edge classification (if supervised)
#         loss_edge = 0
#         if supervised:
#             edge_logits = model.classify_edges(node_embeddings, edge_index)
#             loss_edge = F.cross_entropy(edge_logits, edge_types_tensor)

#         # Combined loss
#         loss = loss_score + (loss_edge if supervised else 0)
#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch} — Loss: {loss.item():.4f} | ScoreLoss: {loss_score.item():.4f}" + (
#                 f" | EdgeLoss: {loss_edge.item():.4f}" if supervised else ""
#             ))

#     return model


# # -----------------------------------------
# # ✅ Extract Embeddings from Trained Model
# # -----------------------------------------
# def extract_embeddings(model, data, edge_index):
#     model.eval()
#     x = torch.tensor(data, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long)
#     with torch.no_grad():
#         node_embeddings, _ = model(x, edge_index)
#     return node_embeddings.cpu().numpy()


# # -----------------------------------------
# # ✅ Cluster Embeddings into Groups (Classes)
# # -----------------------------------------
# def cluster_embeddings(embeddings, n_classes=10):
#     kmeans = KMeans(n_clusters=n_classes, random_state=42)
#     return kmeans.fit_predict(embeddings)





## version 1
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # -------------------------
# # Dual-Head Graph Attention Network Definition
# # -------------------------
# class DualHeadGAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, emb_dim, out_dim, heads=4, dropout=0.3):
#         super(DualHeadGAT, self).__init__()
#         # Shared layers
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)

#         # Branch 1: Embedding head (for clustering)
#         self.gat_embed = GATConv(hidden_channels * heads, emb_dim, heads=1, concat=False, dropout=dropout)

#         # Branch 2: Score prediction head
#         self.gat_score = GATConv(hidden_channels * heads, out_dim, heads=1, concat=False, dropout=dropout)

#     def forward(self, x, edge_index):
#         # Shared base attention layer
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.dropout(x, training=self.training)

#         # Two output branches
#         emb_out = self.gat_embed(x, edge_index)  # Embeddings for clustering
#         score_out = self.gat_score(x, edge_index)  # Predicted exam scores (or latent performance)
#         return emb_out, score_out


# # -------------------------
# # Feature Extraction and Transformation from Student Dataset
# # -------------------------
# def extract_features(df: pd.DataFrame):
#     # Select 10 meaningful attributes influencing academic performance
#     features = [
#         'Motivation_Level',
#         'Peer_Influence_Positive',
#         'Peer_Influence_Negative',
#         'Attendance',
#         'Parental_Involvement',
#         'Sleep_Hours',
#         'Extracurricular_Activities',
#         'Tutoring_Sessions',
#         'Internet_Access',
#         'School_Type_Private'
#     ]
#     # Extract and scale the features
#     selected_data = df[features].copy()
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(selected_data)
#     return scaled_data  # Shape: (num_students, 10)


# # -------------------------
# # Clustering Students Using GAT Embeddings
# # -------------------------
# def cluster_students(embeddings, n_classes=10, method="kmeans"):
#     if method == "kmeans":
#         # Use KMeans to cluster students based on GAT embeddings
#         kmeans = KMeans(n_clusters=n_classes, random_state=42)
#         labels = kmeans.fit_predict(embeddings)
#     else:
#         raise NotImplementedError("Only KMeans implemented.")
#     return labels


# # -------------------------
# # Dual-Head GAT Model Training & Allocation Pipeline
# # -------------------------
# def allocate_students_dual_head(data, edge_index, in_dim, num_classes=10):
#     # Initialize Dual-Head GAT model
#     model = DualHeadGAT(in_channels=in_dim, hidden_channels=64, emb_dim=32, out_dim=1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Convert input data and graph edge list to tensors
#     x = torch.tensor(data, dtype=torch.float)
#     edge_index = torch.tensor(edge_index, dtype=torch.long)

#     # Train the Dual-Head GAT model (unsupervised + optional supervised if ground truth exists)
#     model.train()
#     for epoch in range(100):
#         optimizer.zero_grad()
#         emb_out, score_out = model(x, edge_index)
#         loss = F.mse_loss(emb_out, x)  # Reconstruction loss using embedding head
#         loss.backward()
#         optimizer.step()

#     # Extract embeddings from trained model
#     model.eval()
#     with torch.no_grad():
#         embeddings, _ = model(x, edge_index)
#         embeddings = embeddings.numpy()

#     # Cluster students using the learned embeddings
#     class_labels = cluster_students(embeddings, n_classes=num_classes)
#     return class_labels, embeddings
