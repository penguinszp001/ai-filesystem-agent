from collections import deque
from agent import agent_executor, run_agent
import os

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

    result = run_agent(full_input, directory=os.getcwd())

    output = result

    # store agent response
    chat_history.append({"role": "assistant", "content": output})

    return output