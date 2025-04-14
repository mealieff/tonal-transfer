import os
import pympi

# Input and output directories
input_dir = "aligned_lamkang"
output_dir = "tsv_transcripts"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".eaf"):  # Process only ELAN files
        path = os.path.join(input_dir, fname)
        eaf = pympi.Elan.Eaf(path)

        tier_names = list(eaf.get_tier_names())  # Convert to list
        if len(tier_names) != 1:
            print(f"Skipping {fname}: expected 1 tier, found {len(tier_names)}")
            continue

        tier = tier_names[0]
        entries = eaf.get_annotation_data_for_tier(tier)

        # Define output path for TSV file
        outname = os.path.splitext(fname)[0] + ".tsv"
        outpath = os.path.join(output_dir, outname)

        with open(outpath, "w", encoding="utf-8") as out:
            for start, end, value in entries:
                if value.strip():  # Skip empty annotations
                    # Write word in language script, repetition, start time, end time, and transcription
                    out.write(f"{value.strip()}\t{value.strip()}\t{start}\t{end}\t{value.strip()}\n")

        print(f"Saved: {outpath}")

