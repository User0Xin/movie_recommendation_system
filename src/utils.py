from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()


def get_neo4j_graph_driver():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(user, password), database="neo4j")
    return driver
