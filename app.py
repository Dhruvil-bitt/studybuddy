"""FinAssist 💰 — Your AI Personal Finance & Wealth Mentor
Built with Streamlit and Google Gemini.

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

# The System Prompt: Financial Assistant Persona (Role, Task, Context, Rules)
SYSTEM_PROMPT = """Role: You are FinAssist, a friendly, practical, and knowledgeable personal finance mentor.

Task: Explain personal finance, budgeting, investing, saving, taxes, and money management concepts in simple terms with one practical real-life example, then check understanding.

Context: Your users are students, young professionals, and beginners learning how to manage money, invest wisely, build savings, and understand financial instruments (SIPs, Mutual Funds, Stocks, Emergency Funds, Compound Interest, Credit Cards, UPI, Taxes).

Rules:
- Keep answers clear and under 150 words unless the user asks for in-depth details.
- Use simple, jargon-free English. If you introduce a financial term (e.g., ROI, Inflation, SIP), define it in one simple sentence.
- Always provide exactly one real-world calculation or relatable example (e.g., managing pocket money, investing ₹500/month, the 50/30/20 budgeting rule).
- Always include a short reminder: "⚠️ Note: For educational purposes, not certified financial advice."
- If you are unsure, say "I'm not sure" and suggest credible financial resources to check.
- If the question is off-topic, politely redirect the conversation back to personal finance and wealth management.
- End every answer with one short question or practical tip to test their understanding.
"""

QUIZ_RULE = """
Finance Quiz mode is ON: after explaining the concept, ask three multiple-choice questions (MCQs) one at a time about saving, budgeting, or investing scenarios.
Wait for the user's answer before revealing whether it was correct and moving on to the next question.
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
            return response.text or "I couldn't generate an answer. Try asking your finance question another way."
        except errors.APIError as error:
            is_last_model = index == len(MODELS) - 1
            if error.code != 429 or is_last_model:
                raise
            continue  # Fallback to next model if quota is exceeded (429)

    raise RuntimeError("Every model either returned or raised an error.")


def friendly_error(error: errors.APIError) -> str:
    """Provide user-friendly error messages for common API errors."""
    if error.code in (400, 401, 403):
        return "Gemini rejected the API key. Please verify the GEMINI_API_KEY in your .env or Streamlit Secrets."
    if error.code == 429:
        return "Too many requests — today's free-tier quota is currently exhausted. Please try again in a bit or use a fresh API key."
    return f"Gemini returned an error ({error.code}). Please try again shortly."


# --- UI Configuration ---------------------------------------------------------

st.set_page_config(page_title="FinAssist - AI Finance Mentor", page_icon="💰", layout="centered")
st.title("💰 FinAssist")
st.caption("Your smart AI guide to budgeting, investing, saving, and financial literacy.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")
    quiz_mode = st.toggle("Finance Quiz Mode", value=False, help="After explaining, FinAssist quizzes you on money management.")
    
    st.divider()
    st.markdown("### 💡 Quick Topics to Ask:")
    st.markdown("- *What is the 50/30/20 budgeting rule?*")
    st.markdown("- *How does compound interest grow my money?*")
    st.markdown("- *What is an emergency fund and how big should it be?*")
    st.markdown("- *Mutual Funds vs Stocks: What's the difference?*")
    st.markdown("- *How do credit scores work?*")
    
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# API Key Validation
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No API key found. Add your GEMINI_API_KEY to the .env file or Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# Chat history stored in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["text"])

# Chat input box
if question := st.chat_input("Ask any finance question (e.g. How does SIP work?)..."):
    # Save & display user message
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate & display assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial insight..."):
            try:
                answer = ask_gemini(client, st.session_state.messages, quiz_mode)
            except errors.APIError as error:
                answer = friendly_error(error)
        st.markdown(answer)

    st.session_state.messages.append({"role": "model", "text": answer})
