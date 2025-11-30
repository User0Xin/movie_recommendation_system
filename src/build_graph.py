import os

import pandas as pd
from neo4j import GraphDatabase
import utils
from dotenv import load_dotenv

from src.config import FILE_PATH

load_dotenv()
# 加载数据
ratings = pd.read_csv(FILE_PATH / "ratings.csv")
movies = pd.read_csv(FILE_PATH / "movies.csv")
# 连接 Neo4j
driver = utils.get_neo4j_graph_driver()

# 清空数据库
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

# 创建 User 节点
with driver.session() as session:
    for uid in ratings['userId'].unique():
        session.run("CREATE (:User {id: $uid})", uid=int(uid))

# 创建 Movie 节点
with driver.session() as session:
    for idx, row in movies.iterrows():
        session.run("CREATE (:Movie {id: $mid, title: $title})",
                    mid=int(row['movieId']), title=row['title'])

# 创建 Genre 节点并连接 Movie-HAS_GENRE->Genre
genres_set = set()
for g in movies['genres']:
    genres_set.update(g.split('|'))

with driver.session() as session:
    for genre in genres_set:
        session.run("CREATE (:Genre {name: $g})", g=genre)

# 创建 Movie-HAS_GENRE 边
with driver.session() as session:
    for idx, row in movies.iterrows():
        movie_id = int(row['movieId'])
        for g in row['genres'].split('|'):
            session.run("""
            MATCH (m:Movie {id:$mid}), (g:Genre {name:$gname})
            CREATE (m)-[:HAS_GENRE]->(g)
            """, mid=movie_id, gname=g)

# 创建 User-RATED->Movie 边
with driver.session() as session:
    for idx, row in ratings.iterrows():
        session.run("""
        MATCH (u:User {id:$uid}), (m:Movie {id:$mid})
        CREATE (u)-[:RATED {rating:$rating}]->(m)
        """, uid=int(row['userId']), mid=int(row['movieId']), rating=float(row['rating']))
