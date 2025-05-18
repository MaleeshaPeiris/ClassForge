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
import pulp



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

def train_model_from_csv(df, num_classes, academic_weight,wellbeing_weight):
    data, _ = process_datasetfile(df, num_classes, academic_weight,wellbeing_weight)
    model = GAT(in_channels=data.x.shape[1], hidden_channels=16, out_channels=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


    # === 9. Train and Save ===
    for epoch in range(1, 101):
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

def process_datasetfile(df, num_classes, academic_weight,wellbeing_weight):

    scaler = MinMaxScaler()
    all_features = df.loc[:, df.columns != 'student_id']
    all_norm_features = scaler.fit_transform(all_features)
    X_processed = pd.DataFrame(all_norm_features)
  
    G = create_graph(df)
    df_labeled = label_students(df,num_classes)

    print(academic_weight,wellbeing_weight)

    if academic_weight == 100 and wellbeing_weight == 0:
        X_processed = X_processed.iloc[:, 2:7]

    if academic_weight == 80 and wellbeing_weight == 20:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 60 and wellbeing_weight == 40:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 40 and wellbeing_weight == 60:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 20 and wellbeing_weight == 80:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 0 and wellbeing_weight == 100:
        X_processed = X_processed.iloc[:, 7:]

    features = X_processed.values
    labels = df_labeled['label'].values

    labels_df = pd.concat([
        pd.DataFrame(labels),
        df['student_id']
    ], axis=1)
    labels_df.columns = ['label','node_id']

    # === Initialize All Nodes ===
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

    # ===  Convert to PyG Data ===
    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])
    data.y = torch.tensor([data.y[i] for i in range(data.num_nodes)], dtype=torch.long)

    # ===  Create train/val/test masks ===
    train_mask = (data.y != -1)  # Only use labeled nodes for training
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)

        # You can randomly split labeled nodes for val/test:
    labeled_indices = train_mask.nonzero(as_tuple=False).view(-1)
    num_labeled = labeled_indices.size(0)

    # Compute split sizes
    num_train = int(0.8 * num_labeled)
    num_val = int(0.1 * num_labeled)

    # Assign masks
    train_mask = torch.zeros_like(train_mask)
    val_mask = torch.zeros_like(val_mask)
    test_mask = torch.zeros_like(test_mask)

    train_mask[labeled_indices[:num_train]] = True
    val_mask[labeled_indices[num_train:num_train + num_val]] = True
    test_mask[labeled_indices[num_train + num_val:]] = True


    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data, df

def process_csvfile(df, academic_weight,wellbeing_weight):
    scaler = MinMaxScaler()
    all_features = df.loc[:, df.columns != 'student_id']
    all_norm_features = scaler.fit_transform(all_features)
    X_processed = pd.DataFrame(all_norm_features)

    G = create_graph(df)

    if academic_weight == 100 and wellbeing_weight == 0:
        X_processed = X_processed.iloc[:, 2:7]

    if academic_weight == 80 and wellbeing_weight == 20:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 60 and wellbeing_weight == 40:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 40 and wellbeing_weight == 60:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 20 and wellbeing_weight == 80:
        X_processed = X_processed.iloc[:, :5]

    if academic_weight == 0 and wellbeing_weight == 100:
        X_processed = X_processed.iloc[:, 7:]


    features = X_processed.values
    # === 3. Initialize All Nodes ===
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['x'] = torch.tensor(features[i], dtype=torch.float)

    data = from_networkx(G)

    return data, G 

def label_students(df, num_classes):

    num_samples = len(df)
    samples_per_class = num_samples // num_classes
    
    # Create labels with equal distribution
    labels = np.tile(np.arange(num_classes), samples_per_class)
    
    # Handle any remainder by randomly adding a few extra labels (optional)
    remainder = num_samples - len(labels)
    if remainder > 0:
        extra_labels = np.random.choice(np.arange(num_classes), remainder, replace=False)
        labels = np.concatenate([labels, extra_labels])
    
    np.random.shuffle(labels)
    df['label'] = labels

    return df

def create_graph(df):

    scaler = MinMaxScaler()
    all_features = df.loc[:, df.columns != 'student_id']
    all_norm_features = scaler.fit_transform(all_features)
    X_processed = pd.DataFrame(all_norm_features)

   # === Find optimal number of clusters using silhouette score ===
    silhouette_scores = []
    K_range = range(2, 10)  # Try 2 to 9 clusters

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        silhouette_scores.append(score)

    # === Choose best k based on max silhouette score ===
    best_k = K_range[np.argmax(silhouette_scores)]
    print(f"Best number of blocks (clusters) for stochastic block model: {best_k}")

    # === Final clustering with best_k ===
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
    # Compute centralities
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    # Add them to each node as attributes
    for n in G.nodes():
        G.nodes[n]['degree_centrality'] = degree_centrality[n]
        G.nodes[n]['betweenness'] = betweenness[n]
        G.nodes[n]['eigenvector'] = eigenvector[n]


    # Sort nodes by centrality score in descending order
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

    # Calculate top 10% count
    top_n = max(1, int(0.10 * len(sorted_nodes)))  # at least 1 node

    # Get top 10% influencer nodes
    top_influencers = [node for node, score in sorted_nodes[:top_n]]
    isolated_at_risk = [node for node, score in sorted_nodes[-top_n:]]

    # Mark them in the graph
    for node in G.nodes():
        G.nodes[node]['is_influencer'] = node in top_influencers
        df['is_influencer'] = node in top_influencers
        G.nodes[node]['is_isolated_at_risk'] = node in isolated_at_risk 
        df['is_isolated_at_risk'] = node in isolated_at_risk

    return G

def optimize_class_allocation(df, num_class, max_gender_dev=2, max_dev=1):

    n_students = len(df)

    # Derive model output score matrix from model_class (one-hot encoding for now)
    model_class = df['allocated_class'].to_numpy()
    model_output = np.zeros((n_students, num_class))
    model_output[np.arange(n_students), model_class] = 1  

    gender = df['gender_code'].to_numpy()
    bully = df['bullying_experience_flag'].to_numpy()
    influencer = df['is_influencer'].to_numpy()

    students_per_class = n_students // num_class
    total_males = np.sum(gender)
    total_bullies = np.sum(bully)
    total_influencers = np.sum(influencer)

    prob = pulp.LpProblem("ClassAllocation", pulp.LpMinimize)

    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_class)] for i in range(n_students)]

    # Objective: minimize deviation from model allocation
    prob += pulp.lpSum((1 - model_output[i][j]) * x[i][j] for i in range(n_students) for j in range(num_class))

    # Constraint: each student in exactly one class
    for i in range(n_students):
        prob += pulp.lpSum(x[i][j] for j in range(num_class)) == 1

    # Constraint: equal class sizes
    for j in range(num_class):
        prob += pulp.lpSum(x[i][j] for i in range(n_students)) == students_per_class

    # Constraint: gender balance
    ideal_males = total_males / num_class
    for j in range(num_class):
        male_count = pulp.lpSum(gender[i] * x[i][j] for i in range(n_students))
        prob += male_count >= ideal_males - max_gender_dev
        prob += male_count <= ideal_males + max_gender_dev

        # Constraint: bully distribution balance
    ideal_bullies = total_bullies / num_class
    for j in range(num_class):
        bully_count = pulp.lpSum(bully[i] * x[i][j] for i in range(n_students))
        prob += bully_count >= ideal_bullies - max_dev
        prob += bully_count <= ideal_bullies + max_dev


    ideal_influencers = total_influencers / num_class
    for j in range(num_class):
        influencer_count = pulp.lpSum(influencer[i] * x[i][j] for i in range(n_students))
        prob += influencer_count >= ideal_influencers - max_dev
        prob += influencer_count <= ideal_influencers + max_dev

    # Solve the problem
    prob.solve()

    # Assign optimal class
    optimal_classes = []
    for i in range(n_students):
        assigned_class = None
        for j in range(num_class):
            if pulp.value(x[i][j]) == 1:
                assigned_class = j
                break
        if assigned_class is None:
            np.random.randint(0, num_class)
        optimal_classes.append(assigned_class)

    df = df.copy()
    df['optimal_class'] = optimal_classes
    return df
