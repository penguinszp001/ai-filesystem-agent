# Minimal Agentic File Processor

This project is a simple Level 2 AI agent built with LangChain and OpenAI.

It demonstrates how an LLM can:
- Read local files
- Summarize content
- Write new files
- Dynamically explore directories
- Use tools with structured inputs

---

## 🚀 Features

### 1. File Interaction
- Read individual files (`read_file`)
- Write output files (`write_file`)
- List directory contents (`list_files`)

### 2. Structured Tool Inputs (Pydantic)
The `write_file` tool uses a schema:
- filename
- content

This avoids parsing errors and improves reliability.

---

### 3. Directory Scanning
The agent can:
- Look inside a folder
- Identify files
- Process them sequentially

Example:
> "Read all files in a folder and summarize them"

---

### 4. Agent Reasoning Logs
The agent runs with:
- `verbose=True`
- `return_intermediate_steps=True`

This allows you to see:
- which tools were used
- what inputs were passed
- what outputs were returned

This is critical for debugging and understanding behavior.

---

## 🧠 Architecture

User Input  
→ LLM decides next step  
→ Tool execution (file read/write/list)  
→ Observation returned  
→ Repeat until task complete  

---

## 📁 Files

- `agent.py` → main agent logic
- `test_openai.py` → API test script
- `requirements.txt`
- `.env` → contains OPEN_AI_API_KEY

---

## ⚙️ Setup

```bash
pip install -r requirements.txt