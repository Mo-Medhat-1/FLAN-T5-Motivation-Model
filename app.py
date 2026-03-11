import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
import html
import re


# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load the PEFT model (LoRA adapter)
    peft_model_id = (
        Path(__file__).resolve().parent
        / "flan_nudge_balanced_best"
        / "flan_nudge_balanced_best"
    )
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# Function to build the prompt (from the original notebook)
def build_prompt(idea, mood="low"):
    mood_line = "mood: high_motivation" if mood == "high" else "mood: low_motivation"
    return (
        f"You are a helpful assistant that generates motivational nudges.\n"
        f"You are given an idea and a mood.\n"
        f"Generate a short, encouraging nudge (18-28 words) that helps the user stay focused and productive.\n"
        f"The nudge should be positive and actionable.\n"
        f"\n"
        f"idea: {idea}\n"
        f"{mood_line}\n"
        f"nudge:"
    )


# Regex for filtering (from the original notebook)
VERBS_RE = r"^(Start|Do|Take|Spend|Focus on|Show up for|Work on|Open|Finish|Keep)\b"
LOW_KEYWORDS_RE = (
    r"(easier later|later|over time|step by step|tomorrow|worth it later|next steps)"
)
HIGH_KEYWORDS_RE = r"(already|easier and easier|paying off|momentum|on the right track|keeps getting easier|compounding results)"


def normalize_text(text):
    return " ".join(text.strip().split())


def passes_mood(text, mood="low"):
    words = len(text.split())
    if not (18 <= words <= 28):
        return False
    if not re.search(VERBS_RE, text):
        return False
    if mood == "low":
        if re.search(LOW_KEYWORDS_RE, text, re.IGNORECASE):
            return True
    elif mood == "high":
        if re.search(HIGH_KEYWORDS_RE, text, re.IGNORECASE):
            return True
    return False


# Function to generate nudge (from the original notebook)
def generate_nudge(idea, mood="low", n_candidates=8, max_rounds=3):
    prompt = build_prompt(idea, mood=mood)
    first_candidate = None

    for _ in range(max_rounds):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=n_candidates,
        )
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for nudge in decoded_outputs:
            cleaned_nudge = normalize_text(nudge)
            if not cleaned_nudge:
                continue

            if first_candidate is None:
                first_candidate = cleaned_nudge

            if passes_mood(cleaned_nudge, mood):
                return cleaned_nudge

    if first_candidate:
        return first_candidate

    return "Sorry, I couldn't generate a suitable nudge for you at the moment. Please try again with a different idea or mood."


# Streamlit UI
st.set_page_config(
    page_title="Motivation Studio", page_icon=":sparkles:", layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

    .stApp {
        font-family: 'Manrope', sans-serif;
        background:
            radial-gradient(circle at 12% 12%, #ffeecf 0%, rgba(255, 238, 207, 0) 35%),
            radial-gradient(circle at 86% 18%, #d8f3f0 0%, rgba(216, 243, 240, 0) 30%),
            linear-gradient(180deg, #f6f8fb 0%, #eef2f6 100%);
    }

    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        color: #f8fafc;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.26);
        animation: rise 450ms ease;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.05rem;
        line-height: 1.1;
        margin-bottom: 0.45rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .hero-sub {
        font-size: 1rem;
        line-height: 1.5;
        color: #dbeafe;
        margin: 0;
    }

    .panel {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #dbe2ea;
        border-radius: 16px;
        padding: 1rem 1.05rem;
        box-shadow: 0 6px 20px rgba(2, 6, 23, 0.08);
        animation: rise 520ms ease;
    }

    .nudge-card {
        border-left: 5px solid #f97316;
        background: #fff7ed;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.5rem;
    }

    .nudge-text {
        font-size: 1.05rem;
        line-height: 1.65;
        color: #7c2d12;
        margin: 0;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 900px) {
        .hero-title { font-size: 1.55rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "last_nudge" not in st.session_state:
    st.session_state.last_nudge = ""

st.markdown(
    """
    <section class="hero-card">
      <div class="hero-title">Motivation Studio</div>
      <p class="hero-sub">
        Turn your current task into a focused, encouraging nudge powered by FLAN-T5 + LoRA.
      </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")
left_col, right_col = st.columns([1.3, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Describe Your Task")

    example_ideas = [
        "Finish the graduation project report introduction.",
        "Review one chapter and take concise notes.",
        "Prepare slides for tomorrow's presentation.",
    ]

    selected_example = st.selectbox(
        "Quick Start (optional)",
        options=["Use your own task"] + example_ideas,
        index=0,
    )

    default_idea = (
        "Start working on my graduation project report."
        if selected_example == "Use your own task"
        else selected_example
    )

    user_idea = st.text_area(
        "What are you working on now?",
        value=default_idea,
        height=120,
        placeholder="Write your current task in one sentence...",
    )

    mood_option = st.radio(
        "Current Motivation Level",
        ("Low Motivation", "High Motivation"),
        horizontal=True,
        index=0,
    )

    # Map mood option to model's expected input
    model_mood = "low" if mood_option == "Low Motivation" else "high"

    if st.button("Generate Nudge", type="primary", use_container_width=True):
        if user_idea and user_idea.strip():
            with st.spinner("Generating your personalized nudge..."):
                try:
                    st.session_state.last_nudge = generate_nudge(user_idea, model_mood)
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
        else:
            st.warning("Please enter an idea or task to generate a nudge.")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Your Nudge")

    if st.session_state.last_nudge:
        safe_nudge = html.escape(st.session_state.last_nudge)
        word_count = len(st.session_state.last_nudge.split())
        st.caption(f"Length: {word_count} words")
        st.markdown(
            f"""
            <div class="nudge-card">
                <p class="nudge-text">{safe_nudge}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Generate your first nudge to see it here.")

    st.markdown("---")
    st.markdown("**How this works**")
    st.markdown(
        "1. You provide a task and motivation level.\n"
        "2. The model creates several candidate nudges.\n"
        "3. The best matching nudge is shown in this panel."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption("Built with FLAN-T5 + LoRA fine-tuning on a custom motivational dataset.")
