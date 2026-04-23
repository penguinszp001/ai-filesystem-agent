from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from pathlib import Path
from dotenv import load_dotenv

from agent import run_agent

app = FastAPI()
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATIONS_BASE_PATH = str(Path(os.getenv("OPERATIONS_BASE_PATH", BASE_DIR)).resolve())

class QueryRequest(BaseModel):
    query: str
    directory: str


@app.get("/")
def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())


@app.post("/query")
def query(req: QueryRequest):

    # Run agent synchronously for now (simple version)
    result = run_agent(req.query, OPERATIONS_BASE_PATH)

    return {
        "status": "done",
        "result": result
    }

@app.get("/info")
def info():
    return {
        "directory": OPERATIONS_BASE_PATH
    }
