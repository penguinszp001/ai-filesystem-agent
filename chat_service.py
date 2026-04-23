from collections import deque
from pathlib import Path
from agent import run_agent
import os
from dotenv import load_dotenv

load_dotenv()

# store last 5 messages
chat_history = deque(maxlen=5)


def run_chat(query: str):
    """
    Handles chat memory + agent execution
    """

    # add user message
    chat_history.append({"role": "user", "content": query})

    # build context string
    context = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in chat_history
    ])

    full_input = f"""
    Here is the recent conversation:
    {context}

    New user request:
    {query}
    """

    configured_base_directory = os.getenv("ACCESSIBLE_FILEPATH", os.getcwd())
    base_directory = str(Path(configured_base_directory).expanduser().resolve())

    result = run_agent(full_input, directory=base_directory)

    output = result

    # store agent response
    chat_history.append({"role": "assistant", "content": output})

    return output
