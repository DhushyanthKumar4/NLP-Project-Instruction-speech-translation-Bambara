# api.py

from fastapi import FastAPI, UploadFile, File
import torch
import json
import numpy as np
import soundfile as sf
from transformers import MBart50TokenizerFast
from model import PackagedModel, preprocess_audio, translate_audio

app = FastAPI(title="Speech Translation API")

# LOAD MODEL ON STARTUP

model = PackagedModel()
model.load_state_dict(torch.load("./model_package/model.pt", map_location="cpu"))
model.eval()

tokenizer = MBart50TokenizerFast.from_pretrained("./model_package")
tokenizer.src_lang = "en_XX"

with open("./model_package/config.json") as f:
    config = json.load(f)

# ENDPOINT

@app.post("/translate-audio")
async def translate_audio_api(file: UploadFile = File(...)):

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes))

    output = translate_audio(
        audio,
        model,
        tokenizer,
        config
    )

    return {"translation": output}
