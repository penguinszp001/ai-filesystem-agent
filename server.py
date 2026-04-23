from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from chat_service import run_chat
import os
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
load_dotenv()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/info")
def info():
    configured_base_directory = os.getenv("ACCESSIBLE_FILEPATH", os.getcwd())
    accessible_directory = str(Path(configured_base_directory).expanduser().resolve())
    return {"directory": accessible_directory}


@app.post("/chat")
def chat(req: QueryRequest):

    result = run_chat(req.query)

    return {
        "status": "done",
        "response": result
    }
