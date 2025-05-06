import os
import json
import librosa
import numpy as np
import re
from praatio import textgrid
from praatio.utilities import errors as praat_errors
from praatio.data_classes.point_tier import PointTier
from praatio.data_classes.interval_tier import IntervalTier


textgrid_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts/Forced_Aligned"
audio_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts"

VOWELS = {"a", "e", "i", "o", "u", "ɪ", "ʊ", "ɛ", "ʌ", "æ", "ɑ", "ɔ", "ə"}

def is_vowel(label):
    return any(v in label.lower() for v in VOWELS)

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

def get_vowel_bounds(word_start, word_end, phoneme_intervals):
    vowels = [
        (intv.start, intv.end)  # Use start and end for start and end times of vowels
        for intv in phoneme_intervals
        if is_vowel(intv.label) and intv.start >= word_start and intv.end <= word_end
    ]
    if not vowels:
        return None, None
    return vowels[0], vowels[-1]

def process_textgrid(textgrid_folder, audio_folder):
    output_data = []

    for file in os.listdir(textgrid_folder):
        if not file.endswith(".TextGrid"):
            continue

        tg_path = os.path.join(textgrid_folder, file)
        audio_path = os.path.join(audio_folder, file.replace(".TextGrid", ".wav"))

        tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
        
        words_tier = tg._tierDict["Word"]
        phoneme_tier = tg._tierDict["Letter"]

        y, sr = librosa.load(audio_path)

        for word_interval in words_tier.entries:
            word = word_interval.label.strip()
            start_time = word_interval.start  # Use start for the start time
            end_time = word_interval.end  # Use end for the end time
            duration = end_time - start_time
            
            if duration == 0:
                print(f"Skipping zero duration interval for word: '{word}' in file {file}")
                continue

            mfcc_features = extract_mfcc(y, sr, start_time, end_time)
            initial_f0 = final_f0 = mid_f0 = 0.0
            mid_time = (start_time + end_time) / 2

            if phoneme_tier:
                first_vowel, last_vowel = get_vowel_bounds(start_time, end_time, phoneme_tier.entries)

                if first_vowel:
                    start_5 = first_vowel[0] + 0.05 * (first_vowel[1] - first_vowel[0])
                    initial_f0 = extract_f0(y, sr, start_5)

                if last_vowel:
                    end_95 = last_vowel[0] + 0.95 * (last_vowel[1] - last_vowel[0])
                    final_f0 = extract_f0(y, sr, end_95)

            mid_f0 = extract_f0(y, sr, mid_time)

            word_data = {
                "word": word,
                "timestamp": {"start": start_time, "end": end_time},
                "initial_f0": initial_f0,
                "final_f0": final_f0,
                "mid_f0": mid_f0,
                "duration": duration,
                "MFCC_features": mfcc_features.tolist()
            }
            output_data.append(word_data)

    return output_data


def save_to_json(data, out_path):
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

output = process_textgrid(textgrid_folder, audio_folder)
save_to_json(output, "lamkang_word_features.json")
print("Saved to lamkang_word_features.json")

