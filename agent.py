import os
import json
import uuid
from dotenv import load_dotenv
import shutil
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
OPERATIONS_BASE_PATH = Path(os.getenv("OPERATIONS_BASE_PATH", ".")).resolve()

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


def resolve_in_base(path: str) -> Path:
    """
    Resolve a user-provided path inside the configured operations base path.
    """
    candidate = Path(path)
    full_path = (OPERATIONS_BASE_PATH / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    try:
        full_path.relative_to(OPERATIONS_BASE_PATH)
    except ValueError:
        raise ValueError(
            f"Path '{path}' is outside allowed base directory: {OPERATIONS_BASE_PATH}"
        )

    return full_path


# -----------------------------
# Tools
# -----------------------------

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    try:
        resolved = resolve_in_base(path)
        with open(resolved, "r") as f:
            content = f.read()

        log_event({
            "type": "tool_result",
            "tool": "read_file",
            "input": {"path": path, "resolved_path": str(resolved)},
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
        resolved = resolve_in_base(filename)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w") as f:
            f.write(content)

        message = f"File '{resolved}' written."

        log_event({
            "type": "tool_call",
            "tool": "write_file",
            "input": {"filename": filename, "resolved_path": str(resolved), "content": content},
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
        resolved = resolve_in_base(directory)
        files = os.listdir(resolved)
        result = "\n".join(files)

        log_event({
            "type": "tool_call",
            "tool": "list_files",
            "input": {"directory": directory, "resolved_path": str(resolved)},
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
        resolved_start = resolve_in_base(start_path)
        for root, dirs, _ in os.walk(resolved_start):
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
            "input": {"name": name, "start_path": start_path, "resolved_start_path": str(resolved_start)},
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
        p = resolve_in_base(path)

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
            "input": {"path": path, "resolved_path": str(p)},
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

        resolved = resolve_in_base(directory)
        for name in os.listdir(resolved):
            full_path = os.path.join(resolved, name)

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
            "input": {"directory": directory, "resolved_path": str(resolved)},
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
        resolved = resolve_in_base(path)
        base64_image = encode_image(str(resolved))

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
            "input": {"path": path, "resolved_path": str(resolved)},
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


@tool
def sort_images_by_type(directory: str) -> str:
    """
    Sort images in a directory into folders by file type (png, jpg, etc).
    """

    try:
        p = resolve_in_base(directory)

        if not p.exists():
            return f"Directory not found: {directory}"

        moved = []

        for file in p.iterdir():
            if file.is_file():
                ext = file.suffix.lower().replace(".", "")

                if ext in ["png", "jpg", "jpeg"]:
                    target_dir = p / ext
                    target_dir.mkdir(exist_ok=True)

                    target_path = target_dir / file.name
                    shutil.copy2(file, target_path)

                    moved.append(f"{file.name} → {ext}/")

        return "\n".join(moved) if moved else "No images found."

    except Exception as e:
        return f"Error: {e}"


@tool
def sort_images_by_content(directory: str, categories: list[str]) -> str:
    """
    Analyze images and sort them into user-defined categories.
    Always includes an 'other' category for low-confidence matches.
    """

    import shutil
    from pathlib import Path

    try:
        if not categories or len(categories) > 4:
            return "Error: You must provide between 1 and 4 categories."

        # Normalize + enforce 'other'
        categories = [c.lower().strip() for c in categories]
        if "other" not in categories:
            categories.append("other")

        p = resolve_in_base(directory)

        if not p.exists():
            return f"Directory not found: {directory}"

        results = []
        category_list = ", ".join([c for c in categories if c != "other"])

        for file in p.iterdir():

            if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            base64_image = encode_image(str(file))

            # 🔥 Ask for structured + confidence output
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
                                f'{{"category": "other", "confidence": <low_value>}}\n'
                                f"Only return JSON."
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

            raw = response.content.strip()

            # 🔒 Safe parsing
            import json
            try:
                parsed = json.loads(raw)
                label = parsed.get("category", "other").lower()
                confidence = float(parsed.get("confidence", 0))
            except:
                label = "other"
                confidence = 0

            # 🔥 Enforce "squishy" behavior
            if label not in categories or label == "other" or confidence < 0.7:
                label = "other"

            target_dir = p / label
            target_dir.mkdir(exist_ok=True)

            target_path = target_dir / file.name
            shutil.copy2(file, target_path)

            results.append(f"{file.name} → {label}/ (confidence: {confidence:.2f})")

        # 👇 Explain behavior to user explicitly
        summary_note = (
            "\n\nNote: Images are only placed into specific categories when the model is confident. "
            "Otherwise, they are placed into the 'other' folder."
        )

        return ("\n".join(results) if results else "No images processed.") + summary_note

    except Exception as e:
        return f"Error: {e}"


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
    Use this instead of calling file_metadata repeatedly.
    You can organize files using tools:
    - sort_images_by_type for file extensions
    - sort_images_by_content for AI-based classification
    Use these tools instead of manually iterating over files.
    When organizing images:
    - Ask the user to specify, or infer categories from the request
    - Pass categories explicitly into sort_images_by_content
    - Do not invent extra categories beyond what the user asked
    - If the user asks for more than 4 categories, request that they reduce it to at most 4
    All file operations must stay inside the configured operations base path.""" ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [
    read_file,
    write_file,
    list_files_with_metadata,
    file_metadata,
    analyze_image,
    find_directory,
    sort_images_by_type,
    sort_images_by_content
]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    return_intermediate_steps=True,
    max_iterations=20
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
        "directory": directory,
        "operations_base_path": str(OPERATIONS_BASE_PATH)
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
