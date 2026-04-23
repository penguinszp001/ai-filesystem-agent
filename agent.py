import json
import mimetypes
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from filesystem_ops import OPERATION_REGISTRY, execute_step

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


llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0,
)

# Operations that are handled directly in this module (not by filesystem_ops).
LLM_OPERATION_NAMES: list[str] = ["inspect_images"]


def _available_operations_text() -> str:
    names = [*OPERATION_REGISTRY.keys(), *LLM_OPERATION_NAMES]
    return "\n".join(f"- {name}" for name in names)


def _parse_plan(raw_text: str) -> dict:
    """Accept plain JSON or fenced JSON and return dict."""
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict) or "steps" not in parsed:
        raise ValueError("Plan must be a JSON object containing a 'steps' list.")

    steps = parsed.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("'steps' must be a list.")

    return parsed


def make_plan(query: str, base_directory: str) -> dict:
    prompt = f"""
You are a planning assistant for deterministic filesystem automation.

Base directory: {base_directory}

Available operations:
{_available_operations_text()}

Task:
Create a concise execution plan for this user request:
{query}

Return ONLY valid JSON (no markdown) in this schema:
{{
  "plan_summary": "short plain-language plan",
  "steps": [
    {{
      "operation": "one available operation name",
      "args": {{"argument_name": "value"}},
      "reason": "why this step is needed"
    }}
  ]
}}

Rules:
- Use only listed operation names.
- Prefer relative paths under the base directory.
- Keep steps minimal and deterministic.
- For make_directory use args: {{"path": "<directory path>"}}.
- For move_file/copy_file/move_directory/copy_directory use args: {{"source_path": "...", "destination_path": "..."}}.
- For find_directory use args: {{"name": "<directory name>"}} and optional {{"path": "<search root>"}}.
- For inspect_images use args: {{"path": "<file-or-directory path>", "question": "<what to detect or describe>"}}.
- inspect_images is read-only and should be used before any image-based file moving decisions.
- Do NOT use aliases like "directory_name"; always emit "name".
- If task is ambiguous or unsafe, return an empty steps list with an explanatory plan_summary.
""".strip()

    response = llm.invoke(prompt)
    return _parse_plan(response.content)


def _add_image_interpretations(results: list[dict]) -> list[dict]:
    """Add concise image interpretations to successful read_file image results."""
    enriched: list[dict] = []

    for result in results:
        cloned = json.loads(json.dumps(result))
        details = cloned.get("details", {})

        if (
            cloned.get("operation") == "read_file"
            and cloned.get("ok")
            and details.get("file_kind") == "image"
            and details.get("data_base64")
        ):
            mime_type = details.get("mime_type", "image/png")
            base64_data = details.get("data_base64")

            response = llm.invoke([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Interpret what is in this image in 1-3 concise sentences. "
                                "Focus only on visible content and avoid categorizing or sorting."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_data}",
                            },
                        },
                    ],
                }
            ])

            details["interpretation"] = response.content
            details["data_base64"] = "[omitted in summary payload]"

        enriched.append(cloned)

    return enriched


def _is_image_file(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(path))
    return bool(mime_type and mime_type.startswith("image/"))


def _inspect_images(base_directory: str, args: dict) -> dict:
    path_arg = args.get("path", ".")
    question = (args.get("question") or "Describe what is visible in the image.").strip()
    resolved = Path(base_directory).resolve() / path_arg if not Path(path_arg).is_absolute() else Path(path_arg).resolve()
    base = Path(base_directory).resolve()

    if not (resolved == base or base in resolved.parents):
        return {
            "operation": "inspect_images",
            "ok": False,
            "message": f"Path '{path_arg}' resolves outside the allowed base directory.",
            "details": {"path": str(resolved), "question": question},
        }

    if not resolved.exists():
        return {
            "operation": "inspect_images",
            "ok": False,
            "message": f"Path not found: {resolved}",
            "details": {"path": str(resolved), "question": question},
        }

    image_paths: list[Path] = []
    if resolved.is_file():
        if _is_image_file(resolved):
            image_paths = [resolved]
    else:
        image_paths = sorted(path for path in resolved.rglob("*") if path.is_file() and _is_image_file(path))

    if not image_paths:
        return {
            "operation": "inspect_images",
            "ok": True,
            "message": "No image files found to inspect.",
            "details": {"path": str(resolved), "question": question, "matches": [], "count": 0},
        }

    analyses: list[dict] = []
    for image_path in image_paths:
        relative_path = str(image_path.relative_to(base))
        read_result = execute_step(base_directory, "read_file", {"path": relative_path})

        if not read_result.get("ok"):
            analyses.append({"path": relative_path, "ok": False, "error": read_result.get("message", "Unable to read image.")})
            continue

        details = read_result.get("details", {})
        mime_type = details.get("mime_type", "image/png")
        base64_data = details.get("data_base64", "")

        response = llm.invoke([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing a local image for filesystem automation. "
                            "Answer with JSON only in this schema: "
                            "{\"summary\": \"1-2 sentence visible description\", "
                            "\"matches_request\": true/false, "
                            "\"evidence\": \"short reason for your decision\"}. "
                            f"Request to evaluate: {question}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    },
                ],
            }
        ])

        try:
            parsed = json.loads(response.content.strip().strip("`").replace("json\n", "", 1))
            analyses.append(
                {
                    "path": relative_path,
                    "ok": True,
                    "summary": parsed.get("summary", ""),
                    "matches_request": bool(parsed.get("matches_request")),
                    "evidence": parsed.get("evidence", ""),
                }
            )
        except Exception:
            analyses.append(
                {
                    "path": relative_path,
                    "ok": True,
                    "summary": str(response.content).strip(),
                    "matches_request": False,
                    "evidence": "Model response was not valid JSON; treated as non-match for safety.",
                }
            )

    matches = [item["path"] for item in analyses if item.get("ok") and item.get("matches_request")]
    return {
        "operation": "inspect_images",
        "ok": True,
        "message": f"Inspected {len(analyses)} image(s); {len(matches)} matched the request.",
        "details": {
            "path": str(resolved),
            "question": question,
            "results": analyses,
            "matches": matches,
            "count": len(analyses),
            "matches_count": len(matches),
        },
    }


def summarize_execution(query: str, plan: dict, results: list[dict]) -> str:
    prompt = f"""
You are a concise assistant summarizing filesystem execution.

Original user request:
{query}

Plan JSON:
{json.dumps(plan, indent=2)}

Execution results JSON:
{json.dumps(results, indent=2)}

Respond with:
1) What was planned.
2) What was executed.
3) Success/failure status per step.
4) Final concise outcome.
""".strip()

    response = llm.invoke(prompt)
    return response.content


def run_agent(query: str, directory: str):
    start_time = time.time()

    log_event({
        "type": "run_start",
        "input": query,
        "directory": directory,
    })

    try:
        plan = make_plan(query=query, base_directory=directory)
        log_event({"type": "plan_created", "plan": plan})
    except Exception as exc:
        error_message = f"Failed to create plan: {exc}"
        log_event({"type": "plan_error", "error": error_message})
        return error_message

    results = []
    for index, step in enumerate(plan.get("steps", []), start=1):
        operation = step.get("operation", "")
        args = step.get("args", {})

        if operation == "inspect_images":
            step_result = _inspect_images(directory, args)
        else:
            step_result = execute_step(directory, operation, args)
        step_result["step_index"] = index
        step_result["reason"] = step.get("reason", "")

        log_event({
            "type": "execution_step",
            "step_index": index,
            "operation": operation,
            "args": args,
            "result": step_result,
        })
        results.append(step_result)

    enriched_results = _add_image_interpretations(results)
    output = summarize_execution(query=query, plan=plan, results=enriched_results)

    duration = time.time() - start_time

    log_event({
        "type": "chat_turn",
        "user_input": query,
        "agent_output": output[:2500],
        "duration_seconds": round(duration, 3),
    })

    log_event({
        "type": "run_end",
        "output": output,
    })

    return output
