# app.py

import streamlit as st
import torch
import json
import numpy as np
from transformers import MBart50TokenizerFast
from model import PackagedModel, preprocess_audio, translate_audio
import soundfile as sf

st.set_page_config(page_title="Speech Translation AI", layout="wide")

st.title(" Speech-to-Text Translation System")
st.markdown("Upload audio → Get translated text using multimodal transformer")

# LOAD MODEL

@st.cache_resource
def load_pipeline():
    model = PackagedModel()
    model.load_state_dict(torch.load("./model_package/model.pt", map_location="cpu"))
    model.eval()

    tokenizer = MBart50TokenizerFast.from_pretrained("./model_package")
    tokenizer.src_lang = "en_XX"

    with open("./model_package/config.json") as f:
        config = json.load(f)

    return model, tokenizer, config

model, tokenizer, config = load_pipeline()

# AUDIO INPUT

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Read audio
    audio, sr = sf.read(uploaded_file)

    if st.button(" Translate"):
        with st.spinner("Processing..."):

            output = translate_audio(
                audio,
                model,
                tokenizer,
                config
            )

        st.success("Translation Complete!")

        st.subheader(" Output Text")
        st.write(output)
