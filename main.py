import uvicorn
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello world"}

@app.get("/all/")
async def all_movies():
    return {"result": "All movies"}


@app.get("/genres/{genre}")
async def genre_movies(genre: str):
    return {"result": f"선택한장르{genre}"}

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)