"""
This script configures the model
"""

from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC,Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
import os
from pathlib import Path
import json

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
LOCAL_MODEL_DIR = Path("./models/xlsr")
VOCAB_FILE = LOCAL_MODEL_DIR / "vocab.json"

def create_vocab():
    vocab_list = list("abcdefghijklmnopqrstuvwxyz ") + ["|", "[UNK]", "[PAD]"]
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
    vocab_dict["|"] = vocab_dict.pop(" ")
    
    VOCAB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    print("vocab.json created.")

# === STEP 2: Download and Save Model Locally ===
def ensure_model_downloaded():
    print(f"Downloading and saving XLS-R model to {LOCAL_MODEL_DIR}")
    
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_FILE,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LOCAL_MODEL_DIR)
    processor.save_pretrained(LOCAL_MODEL_DIR)
    print("Model, tokenizer, and processor saved locally.")

"""
# === this generates a README for the model, not touching unless sending to huggingface ===
def prepare_readme():
    readme_text = """
language: ["lamkang", "unspecified"]
tags:
- audio
- speech
- automatic-speech-recognition
- xls-r
- wav2vec2
- lamkang-language
---
"""
# XLS-R Model for Lamkang STT (Base)

This repository contains a version of the [`facebook/wav2vec2-large-xlsr-53`] model prepared for Lamkang transcription. All data from CoRSAL. 
"""
    readme_file = LOCAL_MODEL_DIR / "README.md"
    readme_file.write_text(readme_text)
    print("README.md written.")
"""

# === for running ===
"""

if __name__ == "__main__":
    create_vocab()
    ensure_model_downloaded()
    prepare_readme()

