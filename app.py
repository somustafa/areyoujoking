import streamlit as st
import os
import torch
import torch.nn.functional as F
import whisper
import requests
from transformers import AutoModelForSequenceClassification, BertTokenizer

# --- MODEL YÜKLƏMƏ ---
@st.cache_resource
def load_models():
    # Sənin Hugging Face model yolun və qovluğun
    model_repo = "sonaamus/areyoujoking"
    sub_folder = "joke_model"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model və Tokenizer-i Hugging Face-dən yükləyirik
    tokenizer = BertTokenizer.from_pretrained(model_repo, subfolder=sub_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo, subfolder=sub_folder).to(device)
    whisper_model = whisper.load_model("base", device=device)
    
    return tokenizer, model, whisper_model, device

# --- TELEGRAM MESAJ ---
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

# --- UI AYARLARI ---
st.set_page_config(page_title="Humor Analyzer", layout="centered")

ASSETS = {
    "GIF": {f"level{i}": f"memes/meme{i}.gif" for i in range(1, 6)},
    "IMAGE": {f"level{i}": f"memes/meme{i}.jpg" for i in range(1, 6)}
}
# İstisna: level5 image webp formatındadırsa
ASSETS["IMAGE"]["level5"] = "memes/meme5.webp"

# Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'final_text' not in st.session_state: st.session_state.final_text = ""
if 'score_100' not in st.session_state: st.session_state.score_100 = 0
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- PROQRAMIN MƏNTİQİ ---
if st.session_state.step == 1:
    st.title("Humor Analyzer")
    method = st.radio("Input Method", ["Text", "Voice"])
    style = st.selectbox("Output Style", ["GIF", "Photo", "Number Only"])
    
    if st.button("Continue"):
        st.session_state.update({"method": method, "style": style, "step": 2})
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Analysis Panel")
    tokenizer, model, whisper_model, device = load_models()

    if st.session_state.method == "Voice":
        audio = st.audio_input("Record your voice")
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
                probs = F.softmax(logits / 0.6, dim=-1) # Balın həssaslığı
                raw_score = probs[0][1].item()
                
                st.session_state.score_100 = int(raw_score * 100)
                st.session_state.analyzed = True
            
            send_to_telegram(st.session_state.final_text, st.session_state.score_100)
        else:
            st.error("No input provided")

    if st.session_state.analyzed:
        score = st.session_state.score_100
        st.markdown(f"### Humor Score: {score}/100")
        
        if st.session_state.style != "Number Only":
            level_idx = min(max(int(score / 20) + 1, 1), 5)
            asset_type = "GIF" if st.session_state.style == "GIF" else "IMAGE"
            path = ASSETS[asset_type][f"level{level_idx}"]
            
            if os.path.exists(path):
                st.image(path)
            else:
                st.warning(f"Resource not found: {path}")

    if st.button("Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()