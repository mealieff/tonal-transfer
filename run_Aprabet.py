import os
import subprocess

# === Config ===
input_dir = "tsv_transcripts"
output_dir = "arpabet_dicts"
arpabet_mapping = "lamkang_to_arpabet.txt"  # glyph → ARPAbet mapping (2-column)

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".tsv"):
        continue

    base_name = os.path.splitext(fname)[0]
    transcription_path = os.path.join(input_dir, fname)

    out_2col = os.path.join(output_dir, f"{base_name}_2col.dict")
    out_3col = os.path.join(output_dir, f"{base_name}_3col.dict")
    bad_words = os.path.join(output_dir, f"{base_name}_bad_words.txt")

    subprocess.run([
        "python", "forced-alignment/makeArpabet.py",
        transcription_path,     # (1) TSV transcription file
        arpabet_mapping,        # (2) glyph→ARPAbet mapping
        "no2ColDict",           # (3) skip loading existing 2-col dict
        "no3ColDict",           # (4) skip loading existing 3-col dict
        out_2col,               # (5) output 2-col dict
        out_3col,               # (6) output 3-col dict
        bad_words               # (7) output list of bad words
    ])

    print(f"Processed: {fname}")

