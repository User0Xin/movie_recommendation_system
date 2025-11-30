from fastapi import FastAPI

from agent import Agent
from src.config import FILE_PATH
from src.fuse_predict import fuse_recommend

app = FastAPI()


@app.post("/recommendMovies")
def get_user_movie_recommendation(uid: str):
    movie_dict = {}
    file_path = FILE_PATH / "movies.dat"
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            fields = line.strip().split('::')
            key = fields[0]
            value = fields[1]  # 其余字段作为value
            movie_dict[key] = value
    recommend_mid = fuse_recommend(uid, 10)
    recommend_movies = []
    for mid in recommend_mid:
        recommend_movies.append(movie_dict[mid])
    return recommend_movies


@app.post("/agentQuery")
def agentQuery(query: str, uid: str):
    agent = Agent(uid)
    return agent.query(query)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8989)
