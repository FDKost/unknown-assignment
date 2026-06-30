# AI Shopping Assistant Agent

This project demonstrates how to build an AI shopping assistant using LangChain and a local LLM.

## Requirements

- Python 3.10+
- A local LLM server running at `http://localhost:1234/v1` (e.g., Ollama, local GPT)

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

The script will:
1. Instantiate a local LLM.
2. Define a `get_price` tool that uses a sub‑agent to generate a realistic price table.
3. Create a main agent that plans a shopping list.
4. Invoke the agent with a sample query.
5. Print the conversation, including tool calls and the final shopping list table.
