"""StudyBuddy 📚 — Build Your First AI Chatbot
A study-tutor chatbot built with Streamlit and Google Gemini.

Run with:  streamlit run app.py
"""

import logging
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Suppress internal SDK info/warning logs
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

# Model fallback list to handle rate limits gracefully
MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

# Prompt 3: The system prompt (Role, Task, Context, Rules)
SYSTEM_PROMPT = """Role: You are StudyBuddy, a friendly and patient tutor.

Task: Explain concepts simply, with one real-world example, then check understanding.

Context: Your users are Indian college students preparing for exams. They may ask
about any subject - engineering, science, maths, programming, or general studies.

Rules:
- Keep answers under 150 words unless the student asks for more detail.
- Use simple English. Avoid jargon; if you must use a term, define it.
- Give exactly one example per concept.
- If you are not sure, say "I'm not sure" and suggest what to check.
- If the question is not about studies, politely bring the conversation back to learning.
- End every answer with one short question to check understanding.
"""

QUIZ_RULE = """
Quiz mode is ON: after explaining, ask three multiple-choice questions one at a time.
Wait for the student's answer before revealing whether it was correct and moving on.
"""


def build_system_prompt(quiz_mode: bool) -> str:
    """Return the system prompt, appending quiz instructions if Quiz Mode is enabled."""
    return SYSTEM_PROMPT + QUIZ_RULE if quiz_mode else SYSTEM_PROMPT


def ask_gemini(client: genai.Client, history: list[dict[str, str]], quiz_mode: bool) -> str:
    """Send conversation history to Gemini and return the model response."""
    contents = [
        types.Content(role=message["role"], parts=[types.Part(text=message["text"])])
        for message in history
    ]
    config = types.GenerateContentConfig(system_instruction=build_system_prompt(quiz_mode))

    for index, model in enumerate(MODELS):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or "I couldn't come up with an answer. Try asking another way."
        except errors.APIError as error:
            is_last_model = index == len(MODELS) - 1
            if error.code != 429 or is_last_model:
                raise
            continue  # Fallback to next model if quota is exceeded (429)

    raise RuntimeError("Every model either returned or raised an error.")


def friendly_error(error: errors.APIError) -> str:
    """Provide user-friendly error messages for common API errors."""
    if error.code in (400, 401, 403):
        return "Gemini rejected the API key. Please verify the GEMINI_API_KEY in your .env file."
    if error.code == 429:
        return "Too many requests — today's free-tier quota is currently exhausted across available models. Please try again later or use a fresh API key."
    return f"Gemini returned an error ({error.code}). Please try again shortly."


# --- UI Configuration (Prompt 4: Polish) --------------------------------------

st.set_page_config(page_title="StudyBuddy", page_icon="📚", layout="centered")
st.title("📚 StudyBuddy")
st.caption("Ask me anything you're studying. I explain, then I check you got it.")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    quiz_mode = st.toggle("Quiz mode", value=False, help="After explaining, StudyBuddy quizzes you with 3 MCQs.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# API Key Validation
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No API key found. Add your GEMINI_API_KEY to the .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# Prompt 2: Chat history stored in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["text"])

# Prompt 1: Chat input box
if question := st.chat_input("What are you studying today?"):
    # Save & display user message
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate & display assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_gemini(client, st.session_state.messages, quiz_mode)
            except errors.APIError as error:
                answer = friendly_error(error)
        st.markdown(answer)

    st.session_state.messages.append({"role": "model", "text": answer})
