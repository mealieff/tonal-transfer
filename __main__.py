# __main__.py

import argparse
import torch
import numpy as np
import os
from pathlib import Path
from train import *
from tonelab.model import Tone2VecModel

def main():
    parser = argparse.ArgumentParser(description="Run Tone2Vec processing and evaluation on Lamkang data.")
    parser.add_argument('--audio_dir', type=str, default='/lamkang_data/Lamkang_aligned_audio_and_transcripts', help='Directory with .wav files')
    parser.add_argument('--align_dir', type=str, default='/lamkang_data/Lamkang_aligned_audio_and_transcripts', help='Directory with .TextGrid alignment files')
    parser.add_argument('--model_weights', type=str, default='tonelab/weights/tone2vec.npy', help='Path to pretrained tone2vec weights (.npy)')
    parser.add_argument('--save_model', type=str, default='models/tonelab_model.pt', help='Where to save the trained model')
    parser.add_argument('--do_train', action='store_true', help='If set, will train a classifier on extracted embeddings')
    parser.add_argument('--do_eval', action='store_true', help='If set, will evaluate WER and CER')
    parser.add_argument('--do_cluster', action='store_true', help='If set, will visualize tone clusters using KMeans')

    args = parser.parse_args()

    if not os.path.exists(args.model_weights):
        raise FileNotFoundError(f"Tone2Vec weights not found at {args.model_weights}")
    
    print("Loading pretrained tone2vec weights...")
    tone2vec_weights = np.load(args.model_weights)

    print(f"Processing files in {args.audio_dir}...")
    embeddings, transcripts = process_files(
        audio_dir=args.audio_dir,
        align_dir=args.align_dir,
        model_name="facebook/wav2vec2-large-xlsr-53"
    )

    if args.do_train:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression

        print("Training simple classifier...")
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, transcripts, test_size=0.2, random_state=42
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        torch.save(clf, args.save_model)
        print(f"Model saved to {args.save_model}")

        if args.do_eval:
            print("Evaluating on test set...")
            y_pred = clf.predict(X_test)
            cer_score, wer_score = compute_cer_wer(y_test, y_pred)
            print(f"CER: {cer_score:.4f}, WER: {wer_score:.4f}")

    if args.do_cluster:
        print("Performing KMeans clustering on embeddings...")
        perform_kmeans_clustering(embeddings)

if __name__ == '__main__':
    main()

"""
Example usage:
python3 -m tonelab --do_cluster
python3 -m tonelab --do_train --do_eval


