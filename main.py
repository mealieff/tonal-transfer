import argparse
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from praatio import tgio
from pathlib import Path
import sys
from tonelab.tone2vec import loading, parse_phonemes, tone_feats, plot

def transcribe_audio_segment(segment_audio, sample_rate, model, device):
    tone_representation = model.encode(segment_audio)
    transcription = model.decode(tone_representation)
    return transcription.lower()


def process_files(audio_dir, align_dir, model_name="facebook/wav2vec2-large-xlsr-53"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Tone2Vec model
    model = Tone2Vec(device=device)
    
    audio_dir = Path(audio_dir)
    align_dir = Path(align_dir)

    for wav_file in audio_dir.glob("*.wav"):
        base_name = wav_file.stem
        textgrid_file = align_dir / f"{base_name}.TextGrid"

        if not textgrid_file.exists():
            print(f"Skipping {base_name} (no alignment file found)")
            continue

        waveform, sr = torchaudio.load(wav_file)
        waveform = waveform.squeeze()

        tg = tgio.openTextgrid(textgrid_file, includeEmptyIntervals=False)
        tier_names = tg.tierNameList
        text_tier_name = tier_names[0]

        segment_tier = tg.tierDict[text_tier_name]
        intervals = segment_tier.entryList

        print(f"\nProcessing {base_name} with {len(intervals)} segments...")

        for idx, (start, end, label) in enumerate(intervals):
            if not label.strip():
                continue

            start_frame = int(start * sr)
            end_frame = int(end * sr)
            segment = waveform[start_frame:end_frame]

            if len(segment) == 0:
                continue

            transcription = transcribe_audio_segment(segment.numpy(), sr, model, device)
            print(f"[{start:.2f}-{end:.2f}] Ref: {label.strip()} | Hyp: {transcription.strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe aligned audio segments using XLS-R.")
    parser.add_argument("--audio_dir", required=True, help="Path to directory containing .wav files.")
    parser.add_argument("--align_dir", required=True, help="Path to directory containing TextGrid alignment files.")
    args = parser.parse_args()

    process_files(args.audio_dir, args.align_dir)

## command line examples: 
## python3 main.py  --audio_dir lamkang_data/Lamkang_aligned_audio_and_transcripts --align_dir lamkang_data/Lamkang_aligned_audio_and_transcripts/Forced_Aligned


