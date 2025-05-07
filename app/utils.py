import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.utils import from_networkx



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.gat2(x, edge_index)


def train_model_from_csv(df, num_classes,criterion):
    data, _ = process_datasetfile(df, num_classes)
    model = GAT(in_channels=5, hidden_channels=8, out_channels=num_classes)
    print(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    # === 9. Train and Save ===
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(data.x, data.edge_index)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].argmax(dim=1)
            acc = (pred == data.y[mask]).sum() / mask.sum()
            accs.append(acc.item())

        train_acc, val_acc, test_acc = accs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {train_acc:.2f} | Val: {val_acc:.2f} | Test: {test_acc:.2f}")


    torch.save(model.state_dict(), "model.pth")
    return model

def process_datasetfile(df, num_classes):
    # Normalize numerical features for similarity calculation
    scaler = MinMaxScaler()
    numerical_features = df[["SES", "achievement", "psychological_distress","wellbeing","gender_code"]]
    normalized_features = scaler.fit_transform(numerical_features)
    normalized_features = pd.DataFrame(normalized_features)

    scaler = MinMaxScaler()
    X_processed = scaler.fit_transform(normalized_features)

    # === Step 4: Find optimal number of clusters using silhouette score ===
    silhouette_scores = []
    K_range = range(2, 10)  # Try 2 to 9 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        silhouette_scores.append(score)

    # === Step 5: Choose best k based on max silhouette score ===
    best_k = K_range[np.argmax(silhouette_scores)]
    #print(f"Best number of blocks (clusters): {best_k}")

    # === Step 6: Final clustering with best_k ===
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['block'] = final_kmeans.fit_predict(X_processed)




    block_labels = df['block'].values
    n = len(df)
    k = 4  # number of blocks
    P = np.array([[0.8, 0.2, 0.1, 0.1],
                [0.2, 0.7, 0.2, 0.1],
                [0.1, 0.2, 0.6, 0.3],
                [0.1, 0.1, 0.3, 0.5]])
    
    # === Organize students into blocks ===
    blocks = [np.where(block_labels == i)[0] for i in range(k)]
    #block_sizes = [len(b) for b in blocks]
    #print(block_sizes)

    # === Generate adjacency matrix based on SBM ===
    adj_matrix = np.zeros((n, n))

    for i in range(k):
        for j in range(k):
            for u in blocks[i]:
                for v in blocks[j]:
                    if u < v:  # only fill upper triangle (undirected graph)
                        if np.random.rand() < P[i, j]:
                            adj_matrix[u, v] = 1
                            adj_matrix[v, u] = 1

    # === Create graph from adjacency matrix ===
    G = nx.from_numpy_array(adj_matrix)

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)

    df["degree_centrality"] = df.index.map(degree_centrality)
    df["betweenness_centrality"] = df.index.map(betweenness_centrality)
    df["closeness_centrality"] = df.index.map(closeness_centrality)
    df["eigenvector_centrality"] = df.index.map(eigenvector_centrality)

    # Step 1: Sort by achievement
    df_sorted = df.sort_values("achievement", ascending=False).reset_index(drop=True)

    n = len(df)
    top_10 = df_sorted.iloc[:int(0.3 * n)].copy()
    bottom_10 = df_sorted.iloc[-int(0.3 * n):].copy()

    # Step 2: Combine top and bottom
    selected_students = pd.concat([top_10, bottom_10]).reset_index(drop=True)

    # Step 3: Assign them evenly to 5 classes
    selected_students["academic_semi_label"] = np.tile(np.arange(num_classes), len(selected_students) // num_classes + 1)[:len(selected_students)]
    selected_students = selected_students.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Step 4: Initialize all as -1 (unlabeled)
    df["academic_semi_label"] = -1

    # Step 5: Apply labels to selected students
    # Match by a unique column like 'student_id'
    df.loc[df["student_id"].isin(selected_students["student_id"]), "academic_semi_label"] = selected_students["academic_semi_label"].values

    # Combine all features and semi-labels
    combined_features = pd.concat([
        pd.DataFrame(normalized_features),
        df[['student_id','academic_semi_label']]
    ], axis=1)

    #combined_features.to_csv("normalized_data_with_labels.csv", index = False)
    features = combined_features.iloc[:, :5].values
    labels = combined_features['academic_semi_label'].values  # Shape: (200,)
    print(labels)
    # Combine all features
    labels_df = pd.concat([
        pd.DataFrame(labels),
        combined_features['student_id']
    ], axis=1)
    labels_df.columns = ['label','node_id']

    # === 3. Initialize All Nodes ===
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['x'] = torch.tensor(features[i], dtype=torch.float)
        G.nodes[node]['y'] = -1  # -1 means "unlabeled"


    # Set labels only for known ones
    for _, row in labels_df.iterrows():
        node_id = int(row['node_id'])
        if node_id in G.nodes:
            G.nodes[node_id]['y'] = int(row['label'])
        else:
            print(f"Node {node_id} not found in graph.")   

    # === 4. Convert to PyG Data ===
    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])
    data.y = torch.tensor([data.y[i] for i in range(data.num_nodes)], dtype=torch.long)

    # === 5. Create train/val/test masks ===
    train_mask = (data.y != -1)  # Only use labeled nodes for training
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

    # You can randomly split labeled nodes for val/test:
    labeled_indices = train_mask.nonzero(as_tuple=False).view(-1)
    val_mask[labeled_indices[:30]] = True
    test_mask[labeled_indices[30:60]] = True
    train_mask[val_mask | test_mask] = False  # Remove from training

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data, df

def process_csvfile(df):
    # Normalize numerical features for similarity calculation
    scaler = MinMaxScaler()
    numerical_features = df[["SES", "achievement", "psychological_distress","wellbeing","gender_code"]]
    normalized_features = scaler.fit_transform(numerical_features)
    normalized_features = pd.DataFrame(normalized_features)

    scaler = MinMaxScaler()
    X_processed = scaler.fit_transform(normalized_features)

    # === Step 4: Find optimal number of clusters using silhouette score ===
    silhouette_scores = []
    K_range = range(2, 10)  # Try 2 to 9 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        silhouette_scores.append(score)

    # === Step 5: Choose best k based on max silhouette score ===
    best_k = K_range[np.argmax(silhouette_scores)]
    #print(f"Best number of blocks (clusters): {best_k}")

    # === Step 6: Final clustering with best_k ===
    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['block'] = final_kmeans.fit_predict(X_processed)


    block_labels = df['block'].values
    n = len(df)
    k = 4  # number of blocks
    P = np.array([[0.8, 0.2, 0.1, 0.1],
                [0.2, 0.7, 0.2, 0.1],
                [0.1, 0.2, 0.6, 0.3],
                [0.1, 0.1, 0.3, 0.5]])
    
    # === Organize students into blocks ===
    blocks = [np.where(block_labels == i)[0] for i in range(k)]
    #block_sizes = [len(b) for b in blocks]
    #print(block_sizes)

    # === Generate adjacency matrix based on SBM ===
    adj_matrix = np.zeros((n, n))

    for i in range(k):
        for j in range(k):
            for u in blocks[i]:
                for v in blocks[j]:
                    if u < v:  # only fill upper triangle (undirected graph)
                        if np.random.rand() < P[i, j]:
                            adj_matrix[u, v] = 1
                            adj_matrix[v, u] = 1

    # === Create graph from adjacency matrix ===
    G = nx.from_numpy_array(adj_matrix)
    
    features = df.iloc[:, :5].values
    # === 3. Initialize All Nodes ===
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['x'] = torch.tensor(features[i], dtype=torch.float)

    data = from_networkx(G)

    return data  