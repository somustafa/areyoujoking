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

# Page Config
st.set_page_config(page_title="Humor Analyzer", layout="centered")

# Assets Configuration
ASSETS = {
    "GIF": {f"level{i}": f"memes/meme{i}.gif" for i in range(1, 6)},
    "IMAGE": {f"level{i}": f"memes/meme{i}.jpg" for i in range(1, 6)}
}

def send_to_telegram(text, lang, score):
    token = st.secrets.get("TG_TOKEN")
    chat_id = st.secrets.get("TG_CHAT_ID")
    if token and chat_id:
        message = (f"New Entry\n"
                   f"Text: {text}\n"
                   f"Lang: {lang}\n"
                   f"Score: {score:.2f}")
        url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)
        
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    whisper_model = whisper.load_model("base", device=device)
    return tokenizer, model, whisper_model, device

def save_to_database(text, lang, label):
    file_path = 'collected_data.csv'
    new_data = pd.DataFrame([[datetime.now(), text, lang, label]], 
                            columns=['Timestamp', 'Text', 'Language', 'Is_Joke'])
    if not os.path.isfile(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)

# Session State
if 'step' not in st.session_state: st.session_state.step = 1
if 'final_text' not in st.session_state: st.session_state.final_text = ""
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# UI Logic
if st.session_state.step == 1:
    st.title("Humor Analyzer")
    input_lang = st.selectbox("Input Language", ["English", "Azerbaijani"])
    method = st.radio("Input Method", ["Text", "Voice"])
    style = st.selectbox("Output Style", ["GIF", "Photo", "Number Only"])
    
    if st.button("Continue"):
        st.session_state.update({"input_lang": input_lang, "method": method, "style": style, "step": 2})
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("Analysis")
    
    if st.session_state.method == "Voice":
        audio = st.audio_input("Record your voice")
        if audio:
            _, _, whisper_model, _ = load_models()
            with open("temp.wav", "wb") as f: f.write(audio.getbuffer())
            st.session_state.final_text = whisper_model.transcribe("temp.wav")["text"]
            st.info(f"Detected Text: {st.session_state.final_text}")
            if os.path.exists("temp.wav"): os.remove("temp.wav")
    else:
        st.session_state.final_text = st.text_area("Enter text")

    if st.button("Analyze"):
        if st.session_state.final_text:
            tokenizer, model, _, device = load_models()
            text_to_process = st.session_state.final_text
            
            if st.session_state.input_lang == "Azerbaijani":
                text_to_process = GoogleTranslator(source='az', target='en').translate(text_to_process)
            
            inputs = tokenizer(text_to_process, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=-1)
                st.session_state.score = probs[0][1].item()
                st.session_state.analyzed = True
            
            send_to_telegram(st.session_state.final_text, st.session_state.input_lang, st.session_state.score)
        else:
            st.error("No input provided")

    if st.session_state.method == "Voice":
        # DuplicateID xətası olmasın deyə key əlavə etdik
        audio = st.audio_input("Record your voice", key="voice_recorder")
        
        if audio:
            tokenizer, model, whisper_model, device = load_models()
            
            # Səsi müvəqqəti fayla yazmaq
            temp_path = "temp_recording.wav"
            with open(temp_path, "wb") as f:
                f.write(audio.getbuffer())
            
            try:
                with st.spinner("Converting voice to text..."):
                    # 1. Səsi mətnə çevir
                    result = whisper_model.transcribe(temp_path)
                    st.session_state.final_text = result["text"]
                    st.info(f"Detected Text: {st.session_state.final_text}")
                
                # 2. Mətn boş deyilsə, dərhal analiz et
                if st.session_state.final_text.strip():
                    process_text = st.session_state.final_text
                    
                    # Azerbaijani üçün tərcümə
                    if st.session_state.input_lang == "Azerbaijani":
                        process_text = GoogleTranslator(source='az', target='en').translate(process_text)
                    
                    # Analiz (BERT)
                    inputs = tokenizer(process_text, return_tensors="pt", truncation=True, padding=True).to(device)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                        probs = F.softmax(logits, dim=-1)
                        st.session_state.score = probs[0][1].item()
                        st.session_state.analyzed = True
                    
                    # 3. Avtomatik Telegram-a göndər (Həm mətni, həm balı)
                    send_to_telegram(st.session_state.final_text, st.session_state.input_lang, st.session_state.score)
                    st.success("Analysis complete and sent to Telegram!")
                    
            except Exception as e:
                st.error(f"Error during processing: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    if st.button("Restart"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()