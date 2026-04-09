import sys
import os
import google.generativeai as genai

# Setup
genai.configure(api_key="REDACTED-GEMINI-KEY")
model = genai.GenerativeModel('gemini-2.5-flash')

# 1. Grab all relevant project files
context_data = []
# Feel free to add other extensions like .csv or .json if you have them!
extensions = ('.md', '.py', '.csv') 

for file in os.listdir('.'):
    if file.endswith(extensions) and file != 'ask_gemini.py':
        try:
            with open(file, 'r', encoding='utf-8') as f:
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
