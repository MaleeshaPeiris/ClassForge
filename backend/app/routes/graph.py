from fastapi import APIRouter
from py2neo import Graph

router = APIRouter()
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

@router.get("/graph")
def get_graph():
    query = """
    MATCH (a:Student)-[r:PEER]->(b:Student)
    RETURN a.id AS source, b.id AS target, r.weight AS weight
    """
    edges = graph.run(query).data()
    
    nodes = graph.run("MATCH (n:Student) RETURN n.id AS id, n.name AS name").data()
    return {"nodes": nodes, "edges": edges}
