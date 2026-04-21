import json
import os
import uuid
from datetime import datetime

RUN_ID = str(uuid.uuid4())
os.makedirs("logs", exist_ok=True)

LOG_FILE = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_ID}.jsonl"


def log_event(event: dict):
    event["timestamp"] = datetime.now().isoformat()
    event["run_id"] = RUN_ID

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")