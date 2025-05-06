# services/neo4j_writer.py

from neo4j import GraphDatabase
import numpy as np

# Configure your Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # 🔐 Replace with your actual password

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def clear_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")

def create_student_nodes(tx, num_nodes):
    for i in range(num_nodes):
        tx.run("CREATE (:Student {id: $id})", id=i)

def create_edges_with_labels(tx, edge_index, edge_types):
    label_map = {0: "CONFLICT", 1: "NEUTRAL", 2: "FRIEND"}
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0][i])
        tgt = int(edge_index[1][i])
        rel_type = label_map[int(edge_types[i])]
        tx.run(f"""
            MATCH (a:Student {{id: $src}}), (b:Student {{id: $tgt}})
            CREATE (a)-[:{rel_type}]->(b)
        """, src=src, tgt=tgt)

def save_graph_to_neo4j(edge_index: np.ndarray, edge_types: np.ndarray):
    with driver.session() as session:
        print("🚨 Clearing previous graph...")
        session.write_transaction(clear_graph)

        print("🧑‍🎓 Creating student nodes...")
        num_nodes = int(edge_index.max()) + 1
        session.write_transaction(create_student_nodes, num_nodes)

        print("🔗 Creating labeled edges...")
        session.write_transaction(create_edges_with_labels, edge_index, edge_types)

    print("✅ Graph successfully stored in Neo4j.")
