import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPEN_AI_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

response = llm.invoke("Say hello in one short sentence.")

print(response.content)