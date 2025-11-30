from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_neo4j_graph_driver():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(user, password), database="neo4j")
    return driver


def get_llm_model():
    model_map = {
        'deepseek': ChatOpenAI(
            model=os.getenv('DEEPSEEK_LLM_MODEL'),
            temperature=os.getenv('TEMPERATURE'),
            max_tokens=os.getenv('MAX_TOKENS'),
            base_url=os.getenv('DEEPSEEK_LLM_BASEURL')
        )
    }
    return model_map[os.getenv('LLM_MODEL')]

def structured_output_parser(response_schemas):
    text = '''
    请从以下文本中，抽取出实体信息，并按json格式输出，json包含首尾的 "```json"和 "```"。
    以下是字段含义和类型，要求输出json中，必须包含下列所有字段: \n
    '''
    for schema in response_schemas:
        text += schema.name + ' 字段，表示: ' + schema.description + ', 类型为: ' + schema.type + '\n'
    return text
