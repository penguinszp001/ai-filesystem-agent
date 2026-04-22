from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from chat_service import run_chat
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/info")
def info():
    return {"directory": BASE_DIR}


@app.post("/chat")
def chat(req: QueryRequest):

    result = run_chat(req.query)

    return {
        "status": "done",
        "response": result
    }