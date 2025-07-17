import requests

def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def generate_strategy_code(prompt_text):
    prompt = (
        "You are a Python financial analyst.\n"
        "Write pure Python Pandas code using the already-loaded DataFrame `df`.\n"
        "Don't include imports, CSV loading, or markdown code blocks.\n"
        "Use `df['Close']` and create a column `df['Signal']` as per the strategy.\n"
        f"\nStrategy Description:\n{prompt_text}"
    )
    return call_llm(prompt)

