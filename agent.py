import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

class WriteFileInput(BaseModel):
    filename: str
    content: str

# ---- Tools ---- #

@tool
def read_file(path: str) -> str:
    """Read a file from disk."""
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"



@tool(args_schema=WriteFileInput)
def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"File '{filename}' written."
    except Exception as e:
        return f"Error: {e}"


# ---- LLM ---- #

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0
)

# ---- Prompt ---- #

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ---- Agent ---- #

tools = [read_file, write_file]

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---- Run ---- #

if __name__ == "__main__":
    result = agent_executor.invoke({
        "input": "Read sample.txt, summarize it in 3 sentences, and save to summary.txt"
    })
    print(result)