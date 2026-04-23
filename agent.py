import json
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
LLM_OPERATION_NAMES = [
    "sort_images_by_content",
    # "classify_images",  # intentionally disabled
]


def _available_operations_text() -> str:
    names = [*OPERATION_REGISTRY.keys(), *LLM_OPERATION_NAMES]
    return "\n".join(f"- {name}" for name in names)


def _pick_category_from_interpretation(interpretation: str, categories: list[str]) -> tuple[str, float]:
    response = llm.invoke(
        f"""
Given this image interpretation:
{interpretation}

Choose the best matching category from this list: {', '.join(categories)}.
Return ONLY valid JSON with this schema:
{{"category": "<one category>", "confidence": <0_to_1>}}

Be conservative with confidence.
""".strip()
    )

    try:
        parsed = json.loads(response.content)
        label = str(parsed.get("category", "other")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
        return label, confidence
    except Exception:
        return "other", 0.0


def run_sort_images_by_content(*, base_directory: str, args: dict) -> dict:
    relative_path = str(args.get("path", "."))
    source_dir = (Path(base_directory) / relative_path).resolve()
    root_dir = Path(base_directory).resolve()

    if not (source_dir == root_dir or root_dir in source_dir.parents):
        return {
            "operation": "sort_images_by_content",
            "ok": False,
            "message": f"Path '{relative_path}' resolves outside the allowed base directory.",
            "details": {"path": str(source_dir)},
        }

    if not source_dir.exists() or not source_dir.is_dir():
        return {
            "operation": "sort_images_by_content",
            "ok": False,
            "message": f"Directory not found: {source_dir}",
            "details": {"path": str(source_dir)},
        }

    raw_categories = args.get("categories") or []
    if not isinstance(raw_categories, list) or not raw_categories:
        return {
            "operation": "sort_images_by_content",
            "ok": False,
            "message": "Provide a non-empty 'categories' list.",
            "details": {"categories": raw_categories},
        }

    categories = [str(c).strip().lower() for c in raw_categories if str(c).strip()]
    if "other" not in categories:
        categories.append("other")

    threshold = float(args.get("confidence_threshold", 0.7))
    mode = str(args.get("mode", "copy")).strip().lower()
    mode = mode if mode in {"copy", "move"} else "copy"
    transfer_operation = "move_file" if mode == "move" else "copy_file"

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files = [file for file in source_dir.iterdir() if file.is_file() and file.suffix.lower() in image_suffixes]

    sorted_results: list[dict] = []
    for index, file in enumerate(files):
        relative_source = str(file.relative_to(root_dir).as_posix())
        read_result = execute_step(base_directory, "read_file", {"path": relative_source})
        enriched = _add_image_interpretations([read_result])[0]

        interpretation = enriched.get("details", {}).get("interpretation", "")
        label, confidence = _pick_category_from_interpretation(interpretation, categories)

        if label not in categories or confidence < threshold:
            label = "other"

        category_dir = str((Path(relative_path) / label).as_posix())
        ensure_dir = execute_step(base_directory, "make_directory", {"path": category_dir})
        if not ensure_dir.get("ok"):
            return {
                "operation": "sort_images_by_content",
                "ok": False,
                "message": f"Failed to create target directory '{category_dir}'.",
                "details": {"error": ensure_dir},
            }

        destination_relative = str((Path(relative_path) / label / file.name).as_posix())
        transfer = execute_step(
            base_directory,
            transfer_operation,
            {
                "source_path": relative_source,
                "destination_path": destination_relative,
            },
        )

        if not transfer.get("ok"):
            return {
                "operation": "sort_images_by_content",
                "ok": False,
                "message": f"Failed to {mode} file '{file.name}'.",
                "details": {"error": transfer},
            }

        item = {
            "file": file.name,
            "category": label,
            "confidence": round(confidence, 4),
            "index": index,
            "interpretation": interpretation,
        }
        sorted_results.append(item)
        log_event({"type": "image_sorted_by_content", **item})

    return {
        "operation": "sort_images_by_content",
        "ok": True,
        "message": f"Sorted {len(sorted_results)} image(s) in {source_dir}",
        "details": {
            "path": str(source_dir),
            "categories": categories,
            "confidence_threshold": threshold,
            "mode": mode,
            "results": sorted_results,
            "count": len(sorted_results),
        },
    }


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
- For sort_images_by_content use args: {{"path": "<directory>", "categories": ["cat", "dog"], "confidence_threshold": 0.7}}.
- Do NOT use aliases like "directory_name"; always emit "name".
- If task is ambiguous or unsafe, return an empty steps list with an explanatory plan_summary.
""".strip()

    response = llm.invoke(prompt)
    return _parse_plan(response.content)


def _add_image_interpretations(results: list[dict]) -> list[dict]:
    """Interpret image read results using the LLM while preserving deterministic file ops."""
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
                        {"type": "text", "text": "Describe and interpret this image briefly."},
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

        if operation == "sort_images_by_content":
            step_result = run_sort_images_by_content(base_directory=directory, args=args)
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
