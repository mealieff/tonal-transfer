import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import torch
import torchaudio
import json
import soundfile as sf
from datasets import load_dataset, Dataset
from transformers import (
    AutoFeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction
)
from jiwer import wer, cer

parser = argparse.ArgumentParser(description="Train Wav2Vec2 XLSR with EAF annotations")
parser.add_argument("--data_dir", type=str, default="~/N/project/CoRSAL", help="Path to input data directory")
parser.add_argument("--model_dir", type=str, default="~/tonal-transfer/models", help="Path to training scripts directory")
parser.add_argument("--output_dir", type=str, default="~/tonal-transfer/system_output", help="Path to output transcriptions")
parser.add_argument("--skip_login", action="store_true", help="Opt-out of Hugging Face login")
args = parser.parse_args()

if not args.skip_login:
    from huggingface_hub import notebook_login
    notebook_login()

MODEL_CHECKPOINT = "facebook/wav2vec2-xlsr-53"
BATCH_SIZE = 32
SAMPLING_RATE = 16000

def parse_eaf(eaf_file):
    """
    Parses an ELAN (.eaf) file and extracts aligned transcriptions.
    
    Returns:
        List of dicts with 'start', 'end', and 'text' timestamps.
    """
    tree = ET.parse(eaf_file)
    root = tree.getroot()
    
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_id = ts.attrib["TIME_SLOT_ID"]
        time_value = int(ts.attrib["TIME_VALUE"])  # Time in milliseconds
        time_slots[time_id] = time_value
    
    annotations = []
    for ann in root.findall(".//ANNOTATION"):
        time_ref1 = ann.find(".//ALIGNABLE_ANNOTATION").attrib["TIME_SLOT_REF1"]
        time_ref2 = ann.find(".//ALIGNABLE_ANNOTATION").attrib["TIME_SLOT_REF2"]
        start = time_slots[time_ref1] / 1000  # Convert ms to seconds
        end = time_slots[time_ref2] / 1000
        
        text_element = ann.find(".//ANNOTATION_VALUE")
        if text_element is not None:
            text = text_element.text.strip()
            annotations.append({"start": start, "end": end, "text": text})
    
    return annotations

def extract_audio_segments(wav_file, annotations):
    """
    Extracts segments from a WAV file based on EAF timestamps.
    
    Returns:
        List of dicts with extracted audio & corresponding transcription.
    """
    speech_segments = []
    for ann in annotations:
        start_frame = int(ann["start"] * SAMPLING_RATE)
        end_frame = int(ann["end"] * SAMPLING_RATE)

        speech_array, _ = sf.read(wav_file, start=start_frame, stop=end_frame)
        speech_segments.append({"speech": speech_array, "text": ann["text"]})
    
    return speech_segments

def process_eaf_data(eaf_path, wav_path):
    """
    Parses EAF file and extracts corresponding speech from WAV file.
    
    Returns:
        List of processed training data.
    """
    annotations = parse_eaf(eaf_path)
    return extract_audio_segments(wav_path, annotations)

def create_vocab(data_dir):
    vocab = {}
    eaf_files = [f for f in os.listdir(data_dir) if f.endswith(".eaf")]
    
    for eaf_file in eaf_files:
        annotations = parse_eaf(os.path.join(data_dir, eaf_file))
        for ann in annotations:
            for char in ann["text"]:
                if char not in vocab:
                    vocab[char] = len(vocab)
    
    vocab["<unk>"] = len(vocab)
    vocab["<pad>"] = len(vocab)
    
    vocab_path = os.path.join(args.model_dir, "vocab.json")
    with open(vocab_path, "w") as vocab_file:
        json.dump(vocab, vocab_file)
    
    return vocab_path

# Prepare Dataset
train_data = []
eaf_files = [f for f in os.listdir(args.data_dir) if f.endswith(".eaf")]

for eaf_file in eaf_files:
    wav_file = eaf_file.replace(".eaf", ".wav")
    wav_path = os.path.join(args.data_dir, "audio", wav_file)
    eaf_path = os.path.join(args.data_dir, eaf_file)

    if os.path.exists(wav_path):
        train_data.extend(process_eaf_data(eaf_path, wav_path))

dataset = Dataset.from_list(train_data)

vocab_file = create_vocab(args.data_dir)
processor = Wav2Vec2Processor.from_pretrained(MODEL_CHECKPOINT, vocab_file=vocab_file, padding=True)

def preprocess_data(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=SAMPLING_RATE).input_values[0]
    batch["labels"] = processor(batch["text"]).input_ids
    return batch

dataset = dataset.map(preprocess_data, remove_columns=["speech"])

model = Wav2Vec2ForCTC.from_pretrained(MODEL_CHECKPOINT)

def compute_metrics(pred: EvalPrediction):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    batch_wer = wer(label_str, pred_str)
    batch_cer = cer(label_str, pred_str)

    # Save outputs
    output_path = os.path.join(args.output_dir, "transcriptions.txt")
    with open(output_path, "w") as f:
        for ref, hyp in zip(label_str, pred_str):
            f.write(f"Reference: {ref}\n")
            f.write(f"Prediction: {hyp}\n\n")
    
    return {"wer": batch_wer, "cer": batch_cer}

training_args = TrainingArguments(
    output_dir=args.model_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=500,
    logging_dir=os.path.join(args.model_dir, "logs"),
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Change this if you have a_

