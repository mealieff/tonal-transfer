import os
import subprocess
from glob import glob

# Paths
DATA_DIR = "aligned_lamkang"
OUTPUT_DIR = "alignment_output"
FAVE_ALIGN_PATH = "/path/to/FAVE-align/align.py"  # Update with your actual path
PRAAT_PATH = "/Applications/Praat.app/Contents/MacOS/Praat"  # Adjust for your OS
PRAAT_SCRIPT = "getPhonetics.praat"
FIX_TEXTGRID_SCRIPT = "fixtextgrid.py"
ARPABET_SCRIPT = "makeArpabet.py"
R_SCRIPT = "vowelTriangle.r"

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_make_arpabet(txt_file):
    print(f"Converting to ARPAbet: {txt_file}")
    subprocess.run(["python3", ARPABET_SCRIPT, txt_file], check=True)

def run_fave_align(wav_file, txt_file):
    print(f"Running FAVE-align on: {wav_file}")
    subprocess.run([
        "python3", FAVE_ALIGN_PATH, wav_file, txt_file
    ], check=True)

def run_fix_textgrid(textgrid_file):
    print(f"Fixing TextGrid: {textgrid_file}")
    subprocess.run(["python3", FIX_TEXTGRID_SCRIPT, textgrid_file], check=True)

def run_praat_phonetics(wav_file, textgrid_file):
    print(f"Extracting phonetics from: {textgrid_file}")
    subprocess.run([
        PRAAT_PATH, "--run", PRAAT_SCRIPT, wav_file, textgrid_file
    ], check=True)

def run_vowel_plot():
    print("Generating vowel triangle visualization...")
    subprocess.run(["Rscript", R_SCRIPT], check=True)

def main():
    ensure_output_dir()
    for txt_file in glob(os.path.join(DATA_DIR, "*.txt")):
        base = os.path.splitext(os.path.basename(txt_file))[0]
        wav_file = os.path.join(DATA_DIR, f"{base}.wav")
        textgrid_file = os.path.join(DATA_DIR, f"{base}.TextGrid")

        if not os.path.exists(wav_file):
            print(f"Missing audio file for {base}, skipping.")
            continue

        # Step 1: ARPAbet conversion
        run_make_arpabet(txt_file)

        # Step 2: Forced alignment
        run_fave_align(wav_file, txt_file)

        # Step 3: Fix TextGrid
        if os.path.exists(textgrid_file):
            run_fix_textgrid(textgrid_file)
        else:
            print(f"TextGrid not found for {base}, skipping phonetic extraction.")
            continue

        # Step 4: Extract phonetic data with Praat
        run_praat_phonetics(wav_file, textgrid_file)

    # Step 5: Visualize vowel space
    run_vowel_plot()

if __name__ == "__main__":
    main()

