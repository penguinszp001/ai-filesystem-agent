import json
import os
import time
import uuid
from datetime import datetime

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

def _available_operations_text() -> str:
    return "\n".join(f"- {name}" for name in OPERATION_REGISTRY.keys())


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
