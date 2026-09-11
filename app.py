"""Financial Inclusion Assistant 💰 — Your AI Guide to Financial Literacy & Public Support
Built with Streamlit and Google Gemini.

Run with:  streamlit run app.py
"""

import base64
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Suppress internal SDK info/warning logs
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()


def set_custom_background():
    """Apply a custom 4K background image if present, styled with glassmorphism."""
    # Look for common image formats in the project directory
    image_candidates = ["background.jpg", "background.png", "background.jpeg", "background.webp"]
    found_image = next((img for img in image_candidates if os.path.exists(img)), None)

    if found_image:
        with open(found_image, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(10, 15, 25, 0.82), rgba(10, 15, 25, 0.88)), 
                        url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(15, 23, 42, 0.78) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .stChatMessage {{
            background: rgba(255, 255, 255, 0.06) !important;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 10px;
        }}
        .stChatInputContainer {{
            backdrop-filter: blur(8px);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Model fallback list to handle rate limits gracefully
MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash"]

# The System Prompt: Financial Assistant Persona (Role, Task, Context, Rules)
SYSTEM_PROMPT = """Role: You are the Financial Inclusion Assistant, a friendly, practical, and knowledgeable financial literacy mentor.

Task: Help users understand budgeting, savings, financial literacy, and relevant public-support resources (government schemes, student aid, public relief programs). Avoid pretending to provide professional financial advice; focus on education, awareness, and practical understanding.

Context: Your users are students, young adults, and beginners learning how to manage money, invest wisely, build savings, explore public support resources, and understand financial instruments (SIPs, Emergency Funds, Compound Interest, Banking, UPI, Taxes).

Rules:
- Keep answers clear, supportive, and under 150 words unless more detail is requested.
- Use simple, jargon-free English. Define any financial terms simply in plain language.
- Provide practical examples (e.g., 50/30/20 budgeting rule, emergency fund calculation, public schemes).
- Never pretend to offer certified, licensed, or professional investment/financial advice.
- Always include a short reminder when relevant: "⚠️ Note: For educational purposes only, not professional financial advice."
- Highlight relevant public-support resources or government schemes when users ask for assistance.
- If unsure, say "I'm not sure" and suggest official or credible sources to consult.
- Politely redirect off-topic questions back to budgeting, saving, and financial literacy.
- End every answer with one short question to check understanding.
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

# Check for custom logo image
logo_candidates = ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]
custom_logo = next((img for img in logo_candidates if os.path.exists(img)), None)

st.set_page_config(
    page_title="Financial Inclusion Assistant",
    page_icon=custom_logo if custom_logo else "💰",
    layout="centered",
)
set_custom_background()

# Title and Logo Header
if custom_logo:
    col1, col2 = st.columns([1, 5], vertical_alignment="center")
    with col1:
        st.image(custom_logo, width=70)
    with col2:
        st.title("Financial Inclusion Assistant")
else:
    st.title("💰 Financial Inclusion Assistant")

st.caption("Your smart AI guide to budgeting, savings, financial literacy, and public support resources.")

# Sidebar Controls
with st.sidebar:
    if custom_logo:
        st.image(custom_logo, use_container_width=True)
    st.header("⚙️ Settings")
    quiz_mode = st.toggle("Finance Quiz Mode", value=False, help="After explaining, Financial Inclusion Assistant quizzes you on money management.")
    
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
