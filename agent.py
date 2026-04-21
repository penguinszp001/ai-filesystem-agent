import os
import json
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from datetime import datetime

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

# -----------------------------
# Structured JSON Logger Setup
# -----------------------------

RUN_ID = str(uuid.uuid4())

os.makedirs("logs", exist_ok=True)

log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_ID}.jsonl"


def log_event(event: dict):
    """Write a structured event to JSONL log file."""
    event["timestamp"] = datetime.now().isoformat()
    event["run_id"] = RUN_ID

    with open(log_filename, "a") as f:
        f.write(json.dumps(event) + "\n")


# -----------------------------
# Tools
# -----------------------------

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    try:
        with open(path, "r") as f:
            content = f.read()

        log_event({
            "type": "tool_result",
            "tool": "read_file",
            "input": {"path": path},
            "output": content
        })

        return content

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_result_error",
            "tool": "read_file",
            "input": {"path": path},
            "error": error
        })

        return error


class WriteFileInput(BaseModel):
    filename: str
    content: str


@tool(args_schema=WriteFileInput)
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filename, "w") as f:
            f.write(content)

        message = f"File '{filename}' written."

        log_event({
            "type": "tool_call",
            "tool": "write_file",
            "input": {"filename": filename, "content": content},
            "output": message
        })

        return message

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_call_error",
            "tool": "write_file",
            "input": {"filename": filename, "content": content},
            "error": error
        })

        return error


@tool
def list_files(directory: str) -> str:
    """List all files in a directory."""
    try:
        files = os.listdir(directory)
        result = "\n".join(files)

        log_event({
            "type": "tool_call",
            "tool": "list_files",
            "input": {"directory": directory},
            "output": result
        })

        return result

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_call_error",
            "tool": "list_files",
            "input": {"directory": directory},
            "error": error
        })

        return error


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0
)


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can read files, summarize them, and write outputs."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


# -----------------------------
# Agent
# -----------------------------

tools = [read_file, write_file, list_files]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    query = """
    Look in the folder 'data'.
    Read all text files.
    Summarize each one in 3 sentences.
    Then write a combined summary to 'summarize_files.txt'.
    """

    # Log run start
    log_event({
        "type": "run_start",
        "input": query
    })

    result = agent_executor.invoke({"input": query})

    # Log final output
    log_event({
        "type": "run_end",
        "output": result["output"]
    })

    # Log intermediate reasoning steps
    for step in result["intermediate_steps"]:
        action, observation = step

        log_event({
            "type": "agent_action",
            "tool": action.tool,
            "stage": "planning",
            "input": action.tool_input,
            "output": observation
        })

    print("Run complete.")
    print("Logs saved to:", log_filename)