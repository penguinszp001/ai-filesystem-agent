import os
import json
import uuid
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO

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


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


@tool
def find_directory(name: str, start_path: str = ".") -> str:
    """
    Search for directories by name starting from a base path.
    Returns matching directory paths.
    """

    matches = []

    try:
        for root, dirs, _ in os.walk(start_path):
            for d in dirs:
                if name.lower() in d.lower():
                    full_path = os.path.join(root, d)
                    matches.append(full_path)

        if not matches:
            result = f"No directories found matching '{name}'"
        else:
            result = "\n".join(matches)

        log_event({
            "type": "tool_call",
            "tool": "find_directory",
            "input": {"name": name, "start_path": start_path},
            "output": result
        })

        return result

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_call_error",
            "tool": "find_directory",
            "input": {"name": name, "start_path": start_path},
            "error": error
        })

        return error


@tool
def file_metadata(path: str) -> str:
    """
    Get metadata about a file: size, created time, modified time.
    """
    try:
        p = Path(path)

        if not p.exists():
            return f"File not found: {path}"

        stats = p.stat()

        created = datetime.fromtimestamp(stats.st_ctime)
        modified = datetime.fromtimestamp(stats.st_mtime)
        size = stats.st_size

        result = {
            "path": str(p.resolve()),
            "size_bytes": size,
            "size_readable": format_bytes(size),
            "created": created.isoformat(),
            "modified": modified.isoformat()
        }

        log_event({
            "type": "tool_call",
            "tool": "file_metadata",
            "input": {"path": path},
            "output": result
        })

        return str(result)

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_call_error",
            "tool": "file_metadata",
            "input": {"path": path},
            "error": error
        })

        return error


@tool
def list_files_with_metadata(directory: str) -> str:
    """
    List all files in a directory with metadata (size, modified time).
    """

    try:
        results = []

        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)

            if os.path.isfile(full_path):
                stats = os.stat(full_path)

                results.append({
                    "name": name,
                    "path": full_path,
                    "size_bytes": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
                })

        log_event({
            "type": "tool_call",
            "tool": "list_files_with_metadata",
            "input": {"directory": directory},
            "output": results
        })

        return str(results)

    except Exception as e:
        return f"Error: {e}"


@tool
def analyze_image(path: str) -> str:
    """
    Analyze an image using OpenAI vision model.
    """

    try:
        base64_image = encode_image(path)

        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ])

        log_event({
            "type": "tool_call",
            "tool": "analyze_image",
            "input": {"path": path},
            "output": response.content
        })

        return response.content

    except Exception as e:
        error = f"Error: {e}"

        log_event({
            "type": "tool_call_error",
            "tool": "analyze_image",
            "input": {"path": path},
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


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a file assistant working inside a local filesystem.
    You can:
    - read files
    - list directories
    - search for directories
    - analyze images
    
    Always assume the base directory is the working directory unless otherwise specified.
    When searching, use tools like find_directory.
    You can list files along with metadata using list_files_with_metadata. 
    Use this instead of calling file_metadata repeatedly."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [
    read_file,
    write_file,
    list_files,
    file_metadata,
    analyze_image,
    list_files_with_metadata
]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True
)


# -----------------------------
# MAIN ENTRYPOINT FOR WEB
# -----------------------------

def run_agent(query: str, directory: str):
    """
    Called by web server.
    """

    log_event({
        "type": "run_start",
        "input": query,
        "directory": directory
    })

    result = agent_executor.invoke({
        "input": query
    })

    log_event({
        "type": "run_end",
        "output": result["output"]
    })

    return result["output"]


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