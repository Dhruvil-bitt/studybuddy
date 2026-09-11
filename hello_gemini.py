"""Your first API call to Gemini. Run with: python hello_gemini.py"""

import os
from dotenv import load_dotenv
from google import genai

# Load GEMINI_API_KEY from .env
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain gravity to a 10-year-old in three lines.",
)

print(response.text)
