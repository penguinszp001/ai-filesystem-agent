# Local File Agent UI

A lightweight local AI agent that can read, summarize, and write files
through a simple web interface.

This project combines: 
- LangChain-based tool-using agent 
- FastAPI
backend server 
- Simple browser UI (no frontend build tools) 
- Structured JSONL logging system 
- Local filesystem access

------------------------------------------------------------------------

# What it does

You can open a browser, type a request like:

-   "List all files in <directory>"
-   "Summarize all text files"
-   "Read file1.txt and explain it"
-   "Create a summary file of everything"

The agent will: 
1. Inspect files in the project directory 
2. Read file
contents 
3. Summarize using an LLM 
4. Optionally write output files 
5. Return a response in the browser UI
------------------------------------------------------------------------
# Setup

## 1. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 2. Create `.env` file

Copy `.env.example`

OR

create new `.env` with `OPEN_AI_API_KEY=your_openai_api_key_here`

------------------------------------------------------------------------

## 3. Run the server

``` bash
uvicorn server:app --reload
```

------------------------------------------------------------------------

## 4. Open the UI

http://127.0.0.1:8000

------------------------------------------------------------------------
# Tools

-   read_file
-   write_file
-   list_files

------------------------------------------------------------------------

# Logging

Every run is stored in: `logs/run_<timestamp>_<uuid>.jsonl`

------------------------------------------------------------------------

# Features

-   local execution
-   structured logs
-   simple UI
-   file-based agent