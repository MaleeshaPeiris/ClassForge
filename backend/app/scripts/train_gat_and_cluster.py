# train_gat_and_cluster.py

import sys
import os
import torch
import json
import numpy as np
from collections import Counter

# Add root path for service imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.gat_allocation import (
    train_dual_head_gat,
    extract_embeddings,
    cluster_embeddings
)
from services.edge_builder import build_edge_index_and_types
from services.load_clean_students import load_clean_features

# Constants
TOP_K = 4
NUM_CLASSES = 10
EPOCHS = 100
OUTPUT_PATH = "student_graph_d3.json"

def save_graph_json_with_attention(model, x_tensor, edge_index_tensor, edge_types_tensor, class_labels, path=OUTPUT_PATH):
    """
    Saves student graph with enhanced semantic edge labels, attention weights, and node features.
    """
    model.eval()
    with torch.no_grad():
        # Get node embeddings and classify edges
        node_embeddings, _ = model(x_tensor, edge_index_tensor)
        edge_logits = model.classify_edges(node_embeddings, edge_index_tensor)
        edge_probs = torch.softmax(edge_logits, dim=1).cpu().numpy()
        edge_preds = np.argmax(edge_probs, axis=1)

        # Get attention weights from first GAT layer
        _, (attn_edge_idx, attn_weights) = model.gat1(x_tensor, edge_index_tensor, return_attention_weights=True)
        attn_weights = attn_weights.mean(dim=1).cpu().numpy()

    # === Build Nodes ===
    nodes = [
        {
            "id": i,
            "group": int(class_labels[i]),
            "embedding": emb.cpu().numpy().tolist()
        }
        for i, emb in enumerate(node_embeddings)
    ]

    # === Build Links with Semantic Types ===
    links = []
    for idx, (src, tgt) in enumerate(edge_index_tensor.T.tolist()):
        prob = float(edge_probs[idx][edge_preds[idx]])
        attn = float(attn_weights[idx])

        if prob > 0.5 and attn > 0.02:
            edge_type = "strong_friend"
        elif prob < 0.3 and attn < 0.01:
            edge_type = "conflict"
        else:
            edge_type = "neutral"

        links.append({
            "source": src,
            "target": tgt,
            "type": edge_type,
            "weight": attn,
            "prob": prob
        })

    # Save as D3-compatible JSON
    with open(path, "w") as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2)
    print(f"✅ Saved enhanced D3 graph JSON to: {path}")

def main():
    print("📥 Loading clean features from PostgreSQL...")
    X_np, y_np = load_clean_features()

    print("🔗 Building edge_index and edge_types...")
    edge_index_np, edge_types_np = build_edge_index_and_types(X_np, top_k=TOP_K)

    print("🧠 Training Dual-Head GAT...")
    model = train_dual_head_gat(
        data=X_np,
        labels=y_np,
        edge_index=edge_index_np,
        edge_types=edge_types_np,
        supervised=True,
        epochs=EPOCHS
    )

    print("📤 Extracting embeddings...")
    embeddings = extract_embeddings(model, X_np, edge_index_np)

    print("📊 Clustering student embeddings into classes...")
    class_labels = cluster_embeddings(embeddings, n_classes=NUM_CLASSES)

    print("💾 Saving graph to JSON for D3 visualization...")
    X_tensor = torch.tensor(X_np, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index_np, dtype=torch.long)
    edge_types_tensor = torch.tensor(edge_types_np, dtype=torch.long)
    

    
    save_graph_json_with_attention(
        model, X_tensor, edge_index_tensor, edge_types_tensor, class_labels
    )

    # 📊 Class size distribution
    print("\n📊 Class Size Distribution:")
    for class_id, count in sorted(Counter(class_labels).items()):
        print(f"Class {class_id}: {count} students")

    # 🧾 Allocation summary
    print("\n🧾 Student Allocation:")
    for idx, label in enumerate(class_labels):
        print(f"Student {idx+1:03d} → Class {label}")

if __name__ == "__main__":
    main()






# version 3
# import sys
# import os
# import torch
# import json
# import numpy as np
# from collections import Counter

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from services.gat_allocation import (
#     train_dual_head_gat,
#     extract_embeddings,
#     cluster_embeddings
# )
# from services.edge_builder import build_edge_index_and_types
# from services.load_clean_students import load_clean_features


# def save_graph_json_with_attention(model, x, edge_index, edge_types, class_labels, path="student_graph.json"):
#     """
#     Save graph with semantic edge types, attention weights, and node features for D3.js.
#     """
#     model.eval()
#     with torch.no_grad():
#         node_embeddings, _ = model(x, edge_index)
#         edge_probs = model.classify_edges(node_embeddings, edge_index)
#         edge_probs = torch.softmax(edge_probs, dim=1).detach().cpu().numpy()
#         edge_preds = np.argmax(edge_probs, axis=1)

#         _, (attn_edge_idx, attn_weights) = model.gat1(x, edge_index, return_attention_weights=True)
#         attn_weights = attn_weights.mean(dim=1).detach().cpu().numpy()

#     nodes = []
#     for i, emb in enumerate(node_embeddings):
#         nodes.append({
#             "id": i,
#             "group": int(class_labels[i]),
#             "embedding": emb.detach().cpu().numpy().tolist()
#         })

#     links = []
#     for idx, (src, tgt) in enumerate(edge_index.T.tolist()):
#         prob = float(edge_probs[idx][edge_preds[idx]])
#         attn = float(attn_weights[idx])
#         edge_type = "neutral"
#         if prob > 0.6 and attn > 0.05:
#             edge_type = "strong_friend"
#         elif prob < 0.4 and attn < 0.02:
#             edge_type = "conflict"

#         links.append({
#             "source": src,
#             "target": tgt,
#             "type": edge_type,
#             "weight": attn,
#             "prob": prob
#         })

#     with open(path, "w") as f:
#         json.dump({"nodes": nodes, "links": links}, f, indent=2)

#     print(f"✅ Saved enhanced D3 graph JSON to: {path}")


# def main():
#     print("📥 Loading clean features from PostgreSQL...")
#     X_np, y_np = load_clean_features()
#     X = torch.tensor(X_np, dtype=torch.float)
#     y = torch.tensor(y_np, dtype=torch.float)

#     print("🔗 Building graph edge_index and edge_types...")
#     edge_index_np, edge_types_np = build_edge_index_and_types(X_np, top_k=4)
#     edge_index = torch.tensor(edge_index_np, dtype=torch.long)
#     edge_types = torch.tensor(edge_types_np, dtype=torch.long)

#     print("🧠 Training Dual-Head GAT...")
#     model = train_dual_head_gat(
#         data=X_np,
#         labels=y_np,
#         edge_index=edge_index_np,
#         edge_types=edge_types_np,
#         supervised=True,
#         epochs=100
#     )

#     print("📤 Extracting embeddings from trained model...")
#     embeddings = extract_embeddings(model, X_np, edge_index_np)

#     print("📊 Clustering student embeddings into classes...")
#     class_labels = cluster_embeddings(embeddings, n_classes=10)

#     print("💾 Saving graph to JSON for D3 visualization...")
#     save_graph_json_with_attention(model, X, edge_index, edge_types, class_labels)

#     distribution = Counter(class_labels)
#     print("\n📊 Class Size Distribution:")
#     for class_id, count in sorted(distribution.items()):
#         print(f"Class {class_id}: {count} students")

#     print("\n🧾 Student Allocation:")
#     for idx, label in enumerate(class_labels):
#         print(f"Student {idx+1:03d} → Class {label}")


# if __name__ == "__main__":
#     main()









# import json

# def save_graph_json_d3(full_embeddings, edge_index, edge_probs, attn_weights, class_labels, X, y, path="student_graph_d3.json"):
#     """
#     Save the graph in D3.js-compatible JSON format with semantic edge types and full node attributes.
#     """

#     # === Nodes ===
#     nodes = []
#     for i in range(len(full_embeddings)):
#         nodes.append({
#             "id": i,
#             "group": int(class_labels[i]),
#             "embedding": full_embeddings[i].tolist(),
#             "score": float(y[i]),
#             "motivation": float(X[i][0]),
#             "gpa": float(X[i][6]) if len(X[i]) > 6 else 0.0  # Adjust index if needed
#         })

#     # === Edges ===
#     links = []
#     for idx, (u, v) in enumerate(edge_index.T.tolist()):
#         prob = float(edge_probs[idx])
#         attn = float(attn_weights[idx])

#         # Semantic edge type
#         if prob > 0.6 and attn > 0.05:
#             edge_type = "strong_friend"
#         elif prob < 0.4 and attn < 0.02:
#             edge_type = "conflict"
#         else:
#             edge_type = "neutral"

#         links.append({
#             "source": u,
#             "target": v,
#             "type": edge_type,
#             "weight": attn,
#             "prob": prob
#         })

#     # === Export to JSON ===
#     with open(path, "w") as f:
#         json.dump({"nodes": nodes, "links": links}, f, indent=2)

#     print(f"✅ Exported semantic graph to {path}")






# # train_gat_and_cluster.py

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# import json
# from collections import Counter

# from services.gat_allocation import (
#     train_dual_head_gat,
#     extract_embeddings,
#     cluster_embeddings
# )
# from services.edge_builder import build_edge_index_and_types
# from services.load_clean_students import load_clean_features



# # ✅ Move function definition to top-level
# def save_graph_json(node_embeddings, edge_index, edge_types, class_labels, path="student_graph.json"):
#     nodes = []
#     for i, emb in enumerate(node_embeddings):
#         nodes.append({
#             "id": i,
#             "group": int(class_labels[i]),
#             "embedding": emb.tolist()
#         })

#     links = []
#     for i, (src, tgt) in enumerate(edge_index.T.tolist()):
#         links.append({
#             "source": src,
#             "target": tgt,
#             "type": int(edge_types[i]) if edge_types is not None else 1
#         })

#     with open(path, "w") as f:
#         json.dump({"nodes": nodes, "links": links}, f, indent=2)
#     print(f"✅ Saved graph to {path}")

# def main():
#     print("📥 Loading clean features from PostgreSQL...")
#     X, y = load_clean_features()

#     print("🔗 Building graph edge_index and edge_types...")
#     edge_index, edge_types = build_edge_index_and_types(X, strategy="peer_similarity", top_k=4)

#     print("🧠 Training Dual-Head GAT...")
#     model = train_dual_head_gat(
#         data=X,
#         labels=y,
#         edge_index=edge_index,
#         edge_types=edge_types,
#         supervised=True,
#         epochs=100
#     )

#     print("📤 Extracting embeddings from trained model...")
#     embeddings = extract_embeddings(model, X, edge_index)

#     print("📊 Clustering student embeddings into classes...")
#     class_labels = cluster_embeddings(embeddings, n_classes=10)

#     print("💾 Saving graph to JSON for D3 visualization...")
#     save_graph_json(embeddings, edge_index, edge_types, class_labels)

#     # 📊 Print class distribution
#     distribution = Counter(class_labels)
#     print("\n📊 Class Size Distribution:")
#     for class_id, count in sorted(distribution.items()):
#         print(f"Class {class_id}: {count} students")

#     # 📋 Print allocations
#     print("\n🧾 Student Allocation:")
#     for idx, label in enumerate(class_labels):
#         print(f"Student {idx+1:03d} → Class {label}")

# if __name__ == "__main__":
#     main()







# version 2
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# from services.gat_allocation import (
#     train_dual_head_gat,
#     extract_embeddings,
#     cluster_embeddings
# )
# from services.edge_builder import build_edge_index_and_types
# from services.load_clean_students import load_clean_features

# from collections import Counter

# def main():
#     print("📥 Loading clean features from PostgreSQL...")
#     X, y = load_clean_features()  # X: [N, 10], y: [N]

#     print("🔗 Building graph edge_index and edge_types...")
#     edge_index, edge_types = build_edge_index_and_types(X, strategy="peer_similarity", top_k=4)

#     print("🧠 Training Dual-Head GAT...")
#     model = train_dual_head_gat(
#         data=X,
#         labels=y,
#         edge_index=edge_index,
#         edge_types=edge_types,
#         supervised=True,
#         epochs=100
#     )

#     print("📤 Extracting embeddings from trained model...")
#     embeddings = extract_embeddings(model, X, edge_index)

#     print("📊 Clustering student embeddings into classes...")
#     class_labels = cluster_embeddings(embeddings, n_classes=10)

#     # 📊 Class size distribution
#     distribution = Counter(class_labels)
#     print("\n📊 Class Size Distribution:")
#     for class_id, count in sorted(distribution.items()):
#         print(f"Class {class_id}: {count} students")

#     # 🎯 Student → Class assignment
#     print("\n🧾 Student Allocation:")
#     for idx, label in enumerate(class_labels):
#         print(f"Student {idx+1:03d} → Class {label}")

# if __name__ == "__main__":
#     main()


## version 1
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# from services.gat_allocation import (
#     train_dual_head_gat,
#     extract_embeddings,
#     cluster_embeddings
# )
# from services.edge_builder import build_edge_index_and_types
# from services.load_clean_students import load_clean_features


# def main():
#     print("📥 Loading clean features from PostgreSQL...")
#     X, y = load_clean_features()  # X: [N, 10], y: [N]

#     print("🔗 Building graph edge_index and edge_types...")
#     edge_index, edge_types = build_edge_index_and_types(X, strategy="peer_similarity", top_k=4)

#     print("🧠 Training Dual-Head GAT...")
#     model = train_dual_head_gat(
#         data=X,
#         labels=y,
#         edge_index=edge_index,
#         edge_types=edge_types,
#         supervised=True,  # Set to False if no edge_types
#         epochs=100
#     )

#     print("📤 Extracting embeddings from trained model...")
#     embeddings = extract_embeddings(model, X, edge_index)

#     print("📊 Clustering student embeddings into classes...")
#     class_labels = cluster_embeddings(embeddings, n_classes=10)


#     # Output: student_id to class
#     for idx, label in enumerate(class_labels):
#         print(f"Student {idx+1:03d} → Class {label}")

# if __name__ == "__main__":
#     main()
