from py2neo import Graph, Node, Relationship

graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

def export_to_neo4j(df, edges, embeddings, attention_weights):
    graph.delete_all()

    for i, row in df.iterrows():
        node = Node("Student", id=int(row["id"]), name=row["name"],
                    motivation=float(row["motivation_level"]),
                    embedding=embeddings[i].tolist())
        graph.create(node)

    for source, target, weight in edges:
        a = graph.nodes.match("Student", id=int(source)).first()
        b = graph.nodes.match("Student", id=int(target)).first()
        if a and b:
            edge = Relationship(a, "PEER", b, weight=float(weight))
            graph.create(edge)
