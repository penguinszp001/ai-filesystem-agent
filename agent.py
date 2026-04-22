import os
import json
import uuid
from dotenv import load_dotenv
import shutil
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from datetime import datetime
import base64
from functools import wraps
import time
from PIL import Image
from io import BytesIO


#TODO Right now the agent is:
#TODO
#TODO “LLM + tools”
#TODO
#TODO What you need is:
#TODO
#TODO “LLM proposes → Python enforces → Python verifies → LLM summarizes”

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
# Logging Decorator
# -----------------------------

def logged_tool(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            log_event({
                "type": "tool_start",
                "tool": name,
                "input": kwargs
            })

            try:
                result = func(*args, **kwargs)

                log_event({
                    "type": "tool_end",
                    "tool": name,
                    "input": kwargs,
                    "output": str(result)[:2500]
                })

                return result

            except Exception as e:
                error = f"Error: {e}"

                log_event({
                    "type": "tool_error",
                    "tool": name,
                    "input": kwargs,
                    "error": error
                })

                return error

        return wrapper
    return decorator

# -----------------------------
# Helpers
# -----------------------------

def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def truncate(text, max_len=500):
    return text if len(text) <= max_len else text[:max_len] + "...[truncated]"


# -----------------------------
# Tools
# -----------------------------

@tool
@logged_tool("read_file")
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path, "r") as f:
        return f.read()


class WriteFileInput(BaseModel):
    filename: str
    content: str


@tool(args_schema=WriteFileInput)
@logged_tool("write_file")
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"File '{filename}' written."


@tool
@logged_tool("create_directory")
def create_directory(path: str) -> str:
    """Create a new directory."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return f"Created directory {path}"


@tool
@logged_tool("move_file")
def move_file(source_path: str, destination_path: str) -> str:
    """Move a file from source_path to destination_path."""

    try:
        src = Path(source_path)
        dst = Path(destination_path)

        if not src.exists():
            return f"Source file does not exist: {source_path}"

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))

        return f"Moved file: {src} → {dst}"

    except Exception as e:
        return f"Error moving file: {e}"


@tool
@logged_tool("move_directory")
def move_directory(source_path: str, destination_path: str) -> str:
    """Move a directory from source_path to destination_path."""

    try:
        src = Path(source_path)
        dst = Path(destination_path)

        if not src.exists():
            return f"Source directory does not exist: {source_path}"

        if not src.is_dir():
            return f"Source is not a directory: {source_path}"

        dst.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src), str(dst))

        return f"Moved directory: {src} → {dst}"

    except Exception as e:
        return f"Error moving directory: {e}"


@tool
@logged_tool("verify_directory_state")
def verify_directory_state(directory: str) -> dict:
    """Return actual filesystem state after operations."""
    p = Path(directory)

    return {
        "folders": [x.name for x in p.iterdir() if x.is_dir()],
        "files": [x.name for x in p.iterdir() if x.is_file()],
        "folder_count": len([x for x in p.iterdir() if x.is_dir()]),
        "file_count": len([x for x in p.iterdir() if x.is_file()])
    }


@tool
@logged_tool("list_directory")
def list_directory(directory: str) -> dict:
    """
    List contents of a directory, clearly separating files and folders.
    """

    p = Path(directory)

    if not p.exists():
        return f"Directory not found: {directory}"

    files = []
    folders = []

    for item in p.iterdir():
        if item.is_dir():
            folders.append(item.name)
        else:
            files.append(item.name)

    result = {
        "directory": directory,
        "folders": folders,
        "files": files,
        "folder_count": len(folders),
        "file_count": len(files)
    }

    return result


@tool
@logged_tool("find_directory")
def find_directory(name: str, start_path: str = ".") -> str:
    """
    Search for directories by name starting from a base path.
    Returns matching directory paths.
    """
    matches = []

    for root, dirs, _ in os.walk(start_path):
        for d in dirs:
            if name.lower() in d.lower():
                matches.append(os.path.join(root, d))

    return "\n".join(matches) if matches else f"No directories found matching '{name}'"


@tool
@logged_tool("file_metadata")
def file_metadata(path: str) -> str:
    """Get metadata about a file: size, created time, modified time."""
    p = Path(path)
    stats = p.stat()

    result = {
        "path": str(p.resolve()),
        "size": format_bytes(stats.st_size),
        "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
    }

    return str(result)


@tool
@logged_tool("list_files_with_metadata")
def list_files_with_metadata(directory: str) -> list:
    """List all files in a directory with metadata (size, modified time)."""

    results = []
    files = os.listdir(directory)

    for i, name in enumerate(files):

        full_path = os.path.join(directory, name)

        if os.path.isfile(full_path):
            stats = os.stat(full_path)

            item = {
                "name": name,
                "size": format_bytes(stats.st_size),
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
            }

            results.append(item)

            # 🔥 per-loop logging
            log_event({
                "type": "file_processed",
                "tool": "list_files_with_metadata",
                "index": i,
                "file": name
            })

    return results


@tool
@logged_tool("analyze_image")
def analyze_image(path: str) -> str:
    """Analyze an image using OpenAI vision model."""

    base64_image = encode_image(path)

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ])

    return response.content


@tool
@logged_tool("sort_images_by_type")
def sort_images_by_type(directory: str) -> str:
    """Sort images in a directory into folders by file type (png, jpg, etc)."""

    p = Path(directory)
    moved = []

    files = list(p.iterdir())

    for i, file in enumerate(files):

        if file.is_file():
            ext = file.suffix.lower().replace(".", "")

            if ext in ["png", "jpg", "jpeg"]:

                target_dir = p / ext
                target_dir.mkdir(exist_ok=True)

                shutil.copy2(file, target_dir / file.name)

                moved.append(f"{file.name} → {ext}/")

                log_event({
                    "type": "image_sorted",
                    "tool": "sort_images_by_type",
                    "file": file.name,
                    "category": ext,
                    "index": i
                })

    return "\n".join(moved) if moved else "No images found."


@tool
@logged_tool("sort_images_by_content")
def sort_images_by_content(directory: str, categories: list[str]) -> str:
    """
    Analyze images and sort them into user-defined categories.
    Always includes an 'other' category for low-confidence matches.
    """
    if not categories or len(categories) > 4:
        return "Error: Provide 1–4 categories."

    categories = [c.lower().strip() for c in categories]
    if "other" not in categories:
        categories.append("other")

    p = Path(directory)
    results = []

    category_list = ", ".join([c for c in categories if c != "other"])

    files = list(p.iterdir())

    for i, file in enumerate(files):

        if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        base64_image = encode_image(str(file))

        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Classify this image into ONE of these categories:\n"
                            f"{category_list}\n\n"
                            f"Return a JSON object like:\n"
                            f'{{"category": "<category>", "confidence": <0-1>}}\n\n'
                            f"If the image does not clearly match one category, return:\n"
                            f'{{"category": "other", "confidence": <low_value>}}\n\n'
                            f"Be conservative. Only assign a category if it is a strong match.\n"
                            f"Otherwise, use 'other'.\n\n"
                            f"Only return valid JSON. No explanation."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ])

        try:
            parsed = json.loads(response.content)
            label = parsed.get("category", "other").lower()
            confidence = float(parsed.get("confidence", 0))
        except:
            label = "other"
            confidence = 0

        if label not in categories or confidence < 0.7:
            label = "other"

        target_dir = p / label
        target_dir.mkdir(exist_ok=True)

        shutil.copy2(file, target_dir / file.name)

        results.append(f"{file.name} → {label}/ ({confidence:.2f})")

        # 🔥 per-image logging
        log_event({
            "type": "image_classified",
            "file": file.name,
            "category": label,
            "confidence": confidence,
            "index": i
        })

    return "\n".join(results)


@tool
def get_capabilities() -> str:
    """Return what the agent is allowed to do."""
    return json.dumps({
        "can_read_files": True,
        "can_write_files": True,
        "can_list_directories": True,
        "can_analyze_images": True,
        "can_sort_files_by_type": True,
        "can_sort_files_by_content": True,
        "can_execute_system_commands": False
    })



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
BEFORE doing anything:
 - Always call get_capabilities
 - If the requested action is not supported, stop and inform the user
     
You can:
- read files
- list directories
- search for directories
- analyze images
- create directories and files

Always assume the base directory is the working directory unless otherwise specified.

When searching, use tools like find_directory.

You can list files along with metadata using list_files_with_metadata. 
Use this instead of calling file_metadata repeatedly.

When listing directory contents, use the list_directory tool.
Do not infer folder names from raw file listings.
When using tools that return structured data, never count items manually. If a tool provides a count field, you MUST use it directly.

You can organize files using tools:
- sort_images_by_type for file extensions
- sort_images_by_content for AI-based classification

Use these tools instead of manually iterating over files.

When organizing images:
- Ask the user to specify, or infer categories from the request
- Pass categories explicitly into sort_images_by_content
- Do not invent extra categories beyond what the user asked
- If the user asks for more than 4 categories, request that they reduce it to at most 4

For any operation that modifies files or directories, you MUST verify the result using a follow-up tool call like 
verify_directory_state or returned structured output before responding to the user.
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])


tools = [
    read_file,
    write_file,
    move_file,
    move_directory,
    create_directory,
    verify_directory_state,
    list_directory,
    find_directory,
    file_metadata,
    list_files_with_metadata,
    analyze_image,
    sort_images_by_type,
    sort_images_by_content,
    get_capabilities
]


agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    max_iterations=20
)


# -----------------------------
# Entry Point
# -----------------------------
def run_agent(query: str, directory: str):
    start_time = time.time()

    log_event({
        "type": "run_start",
        "input": query,
        "directory": directory
    })

    max_retries = 3
    attempt = 0
    output = None

    while attempt < max_retries:
        attempt += 1

        log_event({
            "type": "agent_attempt",
            "attempt": attempt,
            "input": query
        })

        # -----------------------------
        # 1. Run main agent
        # -----------------------------
        result = agent_executor.invoke({
            "input": query if attempt == 1 else f"""
Previous attempt failed verification.

Original request:
{query}

Previous result:
{output}

Fix the issue and try again.
"""
        })

        output = result["output"]

        log_event({
            "type": "agent_output_raw",
            "attempt": attempt,
            "output": truncate(output)
        })

        # -----------------------------
        # 2. Verification step
        # -----------------------------
        verification = agent_executor.invoke({
            "input": f"""
You are a filesystem verifier.

Task:
{query}

Agent output:
{output}

Use tools (especially verify_directory_state or list_directory) to confirm correctness.

Return ONLY:
PASS or FAIL + short reason.
"""
        })["output"]

        log_event({
            "type": "verification",
            "attempt": attempt,
            "result": verification
        })

        # -----------------------------
        # 3. Decide whether to retry
        # -----------------------------
        if "PASS" in verification.upper():
            break

    duration = time.time() - start_time

    # 🔥 KEEP EXACTLY YOUR EXISTING LOGGING
    log_event({
        "type": "chat_turn",
        "user_input": query,
        "agent_output": truncate(output),
        "duration_seconds": round(duration, 3)
    })

    log_event({
        "type": "run_end",
        "output": output
    })

    return output