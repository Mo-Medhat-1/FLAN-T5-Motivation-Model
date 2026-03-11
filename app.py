import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from pathlib import Path
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
st.set_page_config(page_title="FLAN-T5 Motivation Nudge Generator", layout="centered")
st.title("🧠 FLAN-T5 Motivation Nudge Generator")
st.markdown(
    "**Generate personalized motivational nudges based on your ideas and mood.**"
)

st.subheader("Your Idea/Task:")
user_idea = st.text_area(
    "Describe what you're working on or thinking about:",
    "Start working on my graduation project report.",
    height=100,
)

st.subheader("Your Current Mood:")
mood_option = st.radio(
    "Select your current motivation level:",
    ("Low Motivation", "High Motivation"),
    index=0,  # Default to Low Motivation
)

# Map mood option to model's expected input
model_mood = "low" if mood_option == "Low Motivation" else "high"

if st.button("Generate Nudge 💪"):
    if user_idea:
        with st.spinner("Generating your personalized nudge..."):
            try:
                nudge = generate_nudge(user_idea, model_mood)
                st.success("Here's your nudge:")
                st.info(f"**{nudge}**")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
    else:
        st.warning("Please enter an idea or task to generate a nudge.")

st.markdown("---")
st.markdown(
    "This project is part of a graduation project showcasing FLAN-T5 fine-tuning with LoRA for personalized motivational text generation. Dataset custom-built from scratch."
)
