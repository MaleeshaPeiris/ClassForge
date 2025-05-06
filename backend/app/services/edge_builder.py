# services/edge_builder.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import random

def build_edge_index_and_types(X, strategy="peer_similarity", top_k=3):
    """
    Build graph edges based on selected strategy.

    Parameters:
    - X: [N, F] feature matrix
    - strategy: 'peer_similarity', 'euclidean', or 'random'
    - top_k: number of neighbors per node

    Returns:
    - edge_index: shape [2, E] where each column is an edge [src, tgt]
    - edge_types: shape [E], labels: 0 = conflict, 1 = neutral, 2 = friend
    """
    num_students = X.shape[0]
    edge_index = []
    edge_types = []

    if strategy == "peer_similarity":
        sim_matrix = cosine_similarity(X)
        np.fill_diagonal(sim_matrix, -1)  # remove self-similarity

        for i in range(num_students):
            top_k_indices = np.argsort(sim_matrix[i])[-top_k:]
            for j in top_k_indices:
                edge_index.append([i, j])
                similarity = sim_matrix[i][j]
                if similarity > 0.8:
                    edge_types.append(2)  # friend
                elif similarity < 0.2:
                    edge_types.append(0)  # conflict
                else:
                    edge_types.append(1)  # neutral

    elif strategy == "euclidean":
        dist_matrix = cdist(X, X, metric="euclidean")
        np.fill_diagonal(dist_matrix, np.inf)  # ignore self

        max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)])
        for i in range(num_students):
            top_k_indices = np.argsort(dist_matrix[i])[:top_k]
            for j in top_k_indices:
                edge_index.append([i, j])
                distance = dist_matrix[i][j]
                if distance < 0.5 * max_dist:
                    edge_types.append(2)  # friend
                elif distance > 0.8 * max_dist:
                    edge_types.append(0)  # conflict
                else:
                    edge_types.append(1)  # neutral

    elif strategy == "random":
        for i in range(num_students):
            choices = random.sample([j for j in range(num_students) if j != i], top_k)
            for j in choices:
                edge_index.append([i, j])
                edge_types.append(random.choice([0, 1, 2]))  # random label

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    return np.array(edge_index).T, np.array(edge_types)



# version2 seem got problem with para when run model

# # services/edge_builder.py

# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def build_edge_index_and_types(X, top_k=3):
#     """
#     Build graph structure for GAT using peer similarity.

#     Parameters:
#     - X: [N, F] matrix of student features (N = number of students)
#     - top_k: number of most similar peers to connect for each student

#     Returns:
#     - edge_index: [2, E] array where each column is an edge (source, target)
#     - edge_types: [E] array of edge labels:
#         0 = conflict (low similarity)
#         1 = neutral (moderate similarity)
#         2 = friend (high similarity)
#     """

#     # Compute cosine similarity between every pair of students
#     sim_matrix = cosine_similarity(X)  # shape: [N, N]

#     # Prevent self-loops by setting diagonal similarity to -1
#     np.fill_diagonal(sim_matrix, -1)

#     # Total number of students (nodes)
#     num_students = X.shape[0]

#     edge_index = []  # list of [source, target] edges
#     edge_types = []  # list of corresponding edge labels

#     # Loop through each student to find their top_k similar peers
#     for i in range(num_students):
#         # Indices of top_k highest similarity values for student i
#         top_k_indices = np.argsort(sim_matrix[i])[-top_k:]

#         # Create directed edges from i → each top_k peer j
#         for j in top_k_indices:
#             edge_index.append([i, j])
#             similarity = sim_matrix[i][j]

#             # Assign edge type based on similarity score
#             if similarity > 0.8:
#                 label = 2  # Friend: Strong similarity
#             elif similarity < 0.2:
#                 label = 0  # Conflict: Very different
#             else:
#                 label = 1  # Neutral: Moderate similarity

#             edge_types.append(label)

#     # Convert edge list to array format required by PyTorch Geometric: [2, E]
#     edge_index = np.array(edge_index).T
#     edge_types = np.array(edge_types)

#     return edge_index, edge_types



# version 1
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def build_edge_index_and_types(X, strategy="peer_similarity", top_k=3):
#     """
#     Returns:
#         edge_index: shape [2, E]
#         edge_types: shape [E] (0=conflict, 1=neutral, 2=friend)
#     """
#     sim_matrix = cosine_similarity(X)  # shape: [N, N]
#     np.fill_diagonal(sim_matrix, -1)   # prevent self-loop
#     num_students = X.shape[0]

#     edge_list = []
#     edge_labels = []

#     for i in range(num_students):
#         top_k_indices = np.argsort(sim_matrix[i])[-top_k:]

#         for j in top_k_indices:
#             edge_list.append([i, j])

#             # Label edges by similarity threshold
#             sim = sim_matrix[i][j]
#             if sim > 0.85:
#                 label = 2  # friend
#             elif sim > 0.6:
#                 label = 1  # neutral
#             else:
#                 label = 0  # conflict

#             edge_labels.append(label)

#     edge_index = np.array(edge_list).T  # shape: [2, E]
#     edge_types = np.array(edge_labels)  # shape: [E]

#     return edge_index, edge_types
