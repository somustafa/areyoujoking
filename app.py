import streamlit as st
import os
import torch
import torch.nn.functional as F
import whisper
import pandas as pd
from datetime import datetime
from transformers import AutoModelForSequenceClassification, BertTokenizer
from deep_translator import GoogleTranslator
import requests

# 1. Configuration
st.set_page_config(page_title="Humor Analyzer", layout="centered")

ASSETS = {
    "GIF": {
        "level1": "memes/meme1.gif",
        "level2": "memes/meme2.gif",
        "level3": "memes/meme3.gif",
        "level4": "memes/meme4.gif",
        "level5": "memes/meme5.gif"
    },
    "IMAGE": {
        "level1": "memes/meme1.jpg",
        "level2": "memes/meme2.jpg",
        "level3": "memes/meme3.jpg",
        "level4": "memes/meme4.jpg",
        "level5": "memes/meme5.webp"
    }
}

def send_to_telegram(text, lang, score):
    token = st.secrets.get("TG_TOKEN")
    chat_id = st.secrets.get("TG_CHAT_ID")
    if token and chat_id:
        level = min(int(score * 5) + 1, 5)
        message = f"New Entry\nText: {text}\nLang: {lang}\nScore: {score:.2f}\nLevel: {level}"
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
        try: requests.get(url)
        except: pass

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    whisper_model = whisper.load_model("base", device=device)
    return tokenizer, model, whisper_model, device

# 2. Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'final_text' not in st.session_state: st.session_state.final_text = ""
if 'score' not in st.session_state: st.session_state.score = 0.0
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# 3. Step 1: Settings
if st.session_state.step == 1:
    st.title("Humor Analyzer")
    input_lang = st.selectbox("Input Language", ["English", "Azerbaijani"])
    method = st.radio("Input Method", ["Text", "Voice"])
    style = st.selectbox("Output Style", ["GIF", "Photo", "Number Only"])
    
    if st.button("Continue"):
        st.session_state.update({"input_lang": input_lang, "method": method, "style": style, "step": 2})
        st.rerun()

# 4. Step 2: Analysis
elif st.session_state.step == 2:
    st.subheader("Analysis Panel")
    tokenizer, model, whisper_model, device = load_models()

    # INPUT SECTION
    if st.session_state.method == "Voice":
        audio = st.audio_input("Record your voice", key="voice_recorder")
        if audio:
            temp_path = "temp_recording.wav"
            with open(temp_path, "wb") as f:
                f.write(audio.getbuffer())
            
            with st.spinner("Converting voice to text..."):
                result = whisper_model.transcribe(temp_path)
                st.session_state.final_text = result["text"]
            
            if os.path.exists(temp_path): os.remove(temp_path)
            st.info(f"Detected Text: {st.session_state.final_text}")
    else:
        st.session_state.final_text = st.text_area("Enter your text here:")

    # ANALYSIS LOGIC
    if st.button("Analyze"):
        if st.session_state.final_text.strip():
            text_to_process = st.session_state.final_text
            
            if st.session_state.input_lang == "Azerbaijani":
                text_to_process = GoogleTranslator(source='az', target='en').translate(text_to_process)
            
            inputs = tokenizer(text_to_process, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                probs = F.softmax(logits / 0.5, dim=-1)
                
                raw_score = probs[0][1].item()
                
                # Süni şaxələndirmə (Əgər model hələ də tənbəllik edirsə):
                # Bu, balın 0.10 və 0.90 arasında daha çox oynamasını təmin edir
                st.session_state.score = raw_score
                st.session_state.analyzed = True
            
            # Send result to Telegram
            send_to_telegram(st.session_state.final_text, st.session_state.input_lang, st.session_state.score)
        else:
            st.error("Please provide input first.")

    # RESULTS DISPLAY
    if st.session_state.analyzed:
        st.divider()
        score = st.session_state.score
        level = min(int(score * 5) + 1, 5)
        
        st.write(f"### Score: {score:.2f}")
        
        if st.session_state.style != "Number Only":
            asset_type = "GIF" if st.session_state.style == "GIF" else "IMAGE"
            path = ASSETS[asset_type][f"level{level}"]
            if os.path.exists(path):
                st.image(path)
            else:
                st.warning("Meme asset not found.")

    if st.button("Restart"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()