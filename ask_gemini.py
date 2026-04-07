import sys
import google.generativeai as genai

# Setup
genai.configure(api_key="REDACTED-GEMINI-KEY")
model = genai.GenerativeModel('gemini-2.5-flash')

# Get piped input + command line argument
piped_input = sys.stdin.read()
user_prompt = sys.argv[1] if len(sys.argv) > 1 else ""

full_prompt = f"{piped_input}\n\nUSER COMMAND: {user_prompt}"

try:
    response = model.generate_content(full_prompt)
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
