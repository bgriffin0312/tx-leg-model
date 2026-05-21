import os
import sys

import google.generativeai as genai

# Load .env into os.environ — tiny inline parser, no external dependency.
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    sys.exit(
        "GEMINI_API_KEY not set. Copy .env.example to .env and paste your key, "
        "or export GEMINI_API_KEY in your shell."
    )

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# 1. Grab all relevant project files
context_data = []
extensions = (".md", ".py", ".csv")

for file in os.listdir("."):
    if file.endswith(extensions) and file != "ask_gemini.py":
        try:
            with open(file, "r", encoding="utf-8") as f:
                context_data.append(f"--- FILE: {file} ---\n{f.read()}\n")
        except Exception as e:
            context_data.append(f"--- FILE: {file} (Error reading: {e}) ---\n")

# 2. Combine files with your question
user_prompt = sys.argv[1] if len(sys.argv) > 1 else "Analyze this project."
full_prompt = "PROJECT CONTEXT:\n" + "\n".join(context_data) + f"\n\nUSER COMMAND: {user_prompt}"

try:
    print(f"Reading {len(context_data)} files... sending to Gemini...")
    response = model.generate_content(full_prompt)
    print("\n--- GEMINI RESPONSE ---")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
