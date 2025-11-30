import json
from fuse_predict import fuse_recommend
from src.config import FILE_PATH
from utils import *
from prompt import *

import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.agents import ZeroShotAgent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import StrOutputParser


class Agent:
    def __init__(self, uid=None):
        load_dotenv()
        self.uid = uid
        self.movie_dict = {}
        file_path = FILE_PATH / "movies.dat"
        with open(file_path, 'r',encoding='latin-1') as f:
            for line in f:
                fields = line.strip().split('::')
                key = fields[0]
                value = fields[1]  # 其余字段作为value
                self.movie_dict[key] = value

    def query(self, query: str):
        # 加载当前图谱中存在的电影类别
        genre_nodes = get_neo4j_graph_driver().session().run("MATCH (n:Genre) RETURN n").data()
        genres = []
        for genre in genre_nodes:
            genres.append(genre['n']['name'])
        # 创建prompt
        genre_prompt = PromptTemplate(
            template=GENRE_PROMPT_TPL,
            input_variables=['query', 'movie_genres']
        )
        # 创建chain
        ner_chain = LLMChain(
            llm=get_llm_model(),
            prompt=genre_prompt,
            verbose=os.getenv('VERBOSE')
        )
        # 使用llm提取出用户问题中的实体
        genre_result = json.loads(ner_chain.run({'query': query, 'movie_genres': genres}))
        print("genre_result:", genre_result)
        graph_query_result = []
        # 遍历找到的实体，去图谱中找
        for genre in genre_result:
            cypher_query = f"""
                                MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {{name: $name}})
                                MATCH ()-[r:RATED]->(m)
                                RETURN 
                                  m.title AS movie_title,
                                  g.name AS genre,
                                  avg(r.rating) AS average_rating,
                                  count(r) AS rating_count
                                ORDER BY average_rating DESC
                                LIMIT 10
                             """
            graph_query_result.append(get_neo4j_graph_driver().session().run(cypher_query, name=genre).data())
        print("graph_query_result:", graph_query_result)

        query_prompt = PromptTemplate.from_template(QUERY_PROMPT_TPL)
        query_chain = LLMChain(
            llm=get_llm_model(),
            prompt=query_prompt,
            verbose=os.getenv('VERBOSE')
        )
        # 3、将两部分信息结合给llm处理
        if graph_query_result == [] and self.uid is not None:
            recommend_mid = fuse_recommend(self.uid, 10)
            for mid in recommend_mid:
                graph_query_result.append(self.movie_dict[mid])
            inputs = {
                'query': query,
                'graph_query_result': '根据当前用户以往评分记录推荐的电影如下：' + str(graph_query_result),
            }
        else:
            inputs = {
                'query': query,
                'graph_query_result': '推荐的电影如下：' + str(graph_query_result)
            }
        return query_chain.run(inputs)


if __name__ == '__main__':
    agent = Agent('1')
    print(agent.query('给我推荐一些电影'))
    # print(agent.query_func("",'美国海关与边境保护局(CBP)干了什么'))
