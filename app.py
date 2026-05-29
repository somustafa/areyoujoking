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

import gdown
import zipfile

@st.cache_resource
def load_models():
    # Məsələn, istifadəçi adın 'sona' və modelin adı 'humor-analyzer'dırsa:
    model_path = "sonaamus/areyoujoking" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    whisper_model = whisper.load_model("base", device=device)
    
    return tokenizer, model, whisper_model, device

# Page Config
st.set_page_config(page_title="Humor Analyzer", layout="centered")

# Assets
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

def send_to_telegram(text, score_100):
    token = st.secrets.get("TG_TOKEN")
    chat_id = st.secrets.get("TG_CHAT_ID")
    if token and chat_id:
        message = (f"New Analysis\n\n"
                   f"Text: {text}\n"
                   f"Humor Score: {score_100}/100")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": message})
        except:
            pass

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    whisper_model = whisper.load_model("base", device=device)
    return tokenizer, model, whisper_model, device

# Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'final_text' not in st.session_state: st.session_state.final_text = ""
if 'score_100' not in st.session_state: st.session_state.score_100 = 0
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# UI Logic
if st.session_state.step == 1:
    st.title("Humor Analyzer")
    # Dil seçimi ləğv edildi, birbaşa English olaraq qeyd olundu
    method = st.radio("Input Method", ["Text", "Voice"])
    style = st.selectbox("Output Style", ["GIF", "Photo", "Number Only"])
    
    if st.button("Continue"):
        st.session_state.update({"input_lang": "English", "method": method, "style": style, "step": 2})
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Analysis Panel")
    tokenizer, model, whisper_model, device = load_models()

    if st.session_state.method == "Voice":
        audio = st.audio_input("Record your voice", key="voice_recorder")
        if audio:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.getbuffer())
            with st.spinner("Processing voice..."):
                result = whisper_model.transcribe("temp_audio.wav")
                st.session_state.final_text = result["text"]
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            st.info(f"Detected Text: {st.session_state.final_text}")
    else:
        st.session_state.final_text = st.text_area("Enter text")

    if st.button("Analyze"):
        if st.session_state.final_text.strip():
            text_to_process = st.session_state.final_text
            
            inputs = tokenizer(text_to_process, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                # 0.6 faktorundan istifadə edərək balları daha kəskin ayırırıq
                probs = F.softmax(logits / 0.6, dim=-1)
                raw_score = probs[0][1].item()
                
                # 0-100 arasına gətirilmə
                st.session_state.score_100 = int(raw_score * 100)
                st.session_state.analyzed = True
            
            send_to_telegram(st.session_state.final_text, st.session_state.score_100)
        else:
            st.error("No input provided")

    if st.session_state.analyzed:
        score = st.session_state.score_100
        st.markdown(f"### Humor Score: {score}/100")
        
        if st.session_state.style != "Number Only":
            level = min(max(int(score / 20) + 1, 1), 5)
            asset_type = "GIF" if st.session_state.style == "GIF" else "IMAGE"
            path = ASSETS[asset_type][f"level{level}"]
            
            if os.path.exists(path):
                st.image(path)
            else:
                st.warning(f"Resource not found: {path}")

    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()