import os
import json
import librosa
import numpy as np
import tgt
import re
from scipy.stats import zscore

textgrid_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts/Forced_Aligned"
audio_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts"

VOWELS = {"a", "e", "i", "o", "u", "ɪ", "ʊ", "ɛ", "ʌ", "æ", "ɑ", "ɔ", "ə"}

def is_vowel(text):
    return any(v in text.lower() for v in VOWELS)

def extract_f0(y, sr, point_time):
    idx = int(point_time * sr)
    window = y[max(0, idx - 256):idx + 256]
    pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
    f0 = np.max(pitches, axis=0)
    return np.mean(f0[np.nonzero(f0)]) if np.any(f0) else 0.0

def extract_mfcc(y, sr, start_time, end_time):
    y_word = y[int(start_time * sr):int(end_time * sr)]
    n_fft = min(2048, len(y_word), 512)  # Ensure n_fft is not larger than the signal length and is at least 512
    mfcc = librosa.feature.mfcc(y=y_word, sr=sr, n_mfcc=13, n_fft=n_fft)
    return mfcc.mean(axis=1)

def get_vowel_bounds(word_start, word_end, phoneme_tier):
    vowels = [
        intv for intv in phoneme_tier
        if is_vowel(intv.text) and intv.start_time >= word_start and intv.end_time <= word_end
    ]

    if vowels:
        first_vowel = vowels[0]
        last_vowel = vowels[-1]
        start_5 = first_vowel.start_time + 0.05 * (first_vowel.end_time - first_vowel.start_time)
        end_5 = last_vowel.start_time + 0.05 * (last_vowel.end_time - last_vowel.start_time)
        return first_vowel, last_vowel
    return None, None


def process_textgrid(textgrid_folder, audio_folder):
    output_data = []

    for file in os.listdir(textgrid_folder):
        if not file.endswith(".TextGrid"):
            continue

        tg_path = os.path.join(textgrid_folder, file)
        audio_path = os.path.join(audio_folder, file.replace(".TextGrid", ".wav"))

        try:
            tg = tgt.io.read_textgrid(tg_path)
        except Exception as e:
            print(f"Error opening TextGrid: {e}")
            continue

        words_tier = tg.get_tier_by_name("Word")  # Access the 'Word' tier
        phoneme_tier = tg.get_tier_by_name("Letter")  # Access the 'Letter' tier

        y, sr = librosa.load(audio_path)

        for word_interval in words_tier:
            print(f"Word Interval: {word_interval}")  # Debugging line to inspect the structure
            word = word_interval.text.strip()
            try:
                start_time = word_interval.start_time  # Access start_time
                end_time = word_interval.end_time  # Access end_time
            except AttributeError:
                print(f"Error: Interval does not have 'start_time' or 'end_time' attribute in {file}")
                continue

            duration = end_time - start_time
            if duration == 0:
                continue

            mfcc_features = extract_mfcc(y, sr, start_time, end_time)
            initial_f0 = final_f0 = mid_f0 = 0.0
            mid_time = (start_time + end_time) / 2

            if phoneme_tier:
                first_vowel, last_vowel = get_vowel_bounds(start_time, end_time, phoneme_tier)

                if first_vowel:
                    start_5 = first_vowel.start_time + 0.05 * (first_vowel.end_time - first_vowel.start_time)
                    initial_f0 = extract_f0(y, sr, start_5)

                if last_vowel:
                    end_95 = last_vowel.start_time + 0.95 * (last_vowel.end_time - last_vowel.start_time)
                    final_f0 = extract_f0(y, sr, end_95)

            mid_f0 = extract_f0(y, sr, mid_time)
            word_data = {
                "word": word,
                "timestamp": {"start": float(start_time), "end": float(end_time)},
                "initial_f0": float(initial_f0),
                "final_f0": float(final_f0),
                "mid_f0": float(mid_f0),
                "duration": float(duration),
                "MFCC_features": [float(x) for x in mfcc_features.tolist()]
                }
            output_data.append(word_data)

    return output_data


def save_to_json(data, out_path):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

output = process_textgrid(textgrid_folder, audio_folder)

### Normalization steps for F0 and MFCCs
initial_f0s = np.array([entry["initial_f0"] for entry in output])
mid_f0s = np.array([entry["mid_f0"] for entry in output])
final_f0s = np.array([entry["final_f0"] for entry in output])
initial_z = zscore(initial_f0s)
mid_z = zscore(mid_f0s)
final_z = zscore(final_f0s)

mfcc_matrix = np.array([entry["MFCC_features"] for entry in output])  # Shape: (N, 13)
mfcc_z = zscore(mfcc_matrix, axis=0)

for i, entry in enumerate(output):
    entry["initial_f0_z"] = float(initial_z[i])
    entry["mid_f0_z"] = float(mid_z[i])
    entry["final_f0_z"] = float(final_z[i])
    entry["MFCC_features_z"] = [float(x) for x in mfcc_z[i].tolist()]

save_to_json(output, "lamkang_word_features.json")
print("Saved to lamkang_word_features.json")


