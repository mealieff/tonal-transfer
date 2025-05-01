## https://github.com/YiYang-github/ToneLab/blob/main/usage.ipynb This script implements this usage guide by ToneLab
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from jiwer import cer, wer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import importlib
import tonelab.tone2vec
importlib.reload(tonelab.tone2vec)
from tonelab.tone2vec import loading, parse_phonemes, tone_feats, plot
from tonelab.model import VGG, ResNet, DenseNet, mlp


def train_and_evaluate(model, train_loader, valid_loader, optimizer, device, num_epochs, save_dir, unique_transcription, print_interval):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_file_path = os.path.join(save_dir, 'training_log.txt')
    unique_transcription = unique_transcription.to(device)
    criterion = nn.L1Loss()

    with open(log_file_path, 'w') as log_file:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        best_valid_accuracy = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_accum = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                values = model(batch_X)
                loss = criterion(values, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()

            scheduler.step()
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_accum / len(train_loader):.4f}\n')

            model.eval()
            train_correct, valid_correct = 0, 0

            with torch.no_grad():
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    values = model(batch_X)
                    train_correct += eval_metric(values, batch_y, unique_transcription, metric_type='acc')

                for batch_X, batch_y in valid_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    values = model(batch_X)
                    valid_correct += eval_metric(values, batch_y, unique_transcription, metric_type='acc')

            valid_accuracy = valid_correct / len(valid_loader.dataset)
            train_accuracy = train_correct / len(train_loader.dataset)

            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n')

            if (epoch + 1) % print_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n')

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                log_file.write(f"Saved Best Model at Epoch {epoch+1} with Valid Accuracy: {best_valid_accuracy:.4f}\n")

            log_file.flush()

def eval_metric(values, batch_y, unique_transcription, metric_type='acc'):
    """
    Calculate the accuracy or MAE of the predictions, after removing superscripts
    from model outputs to match the ground truth format.

    Args:
    - values: torch.tensor of shape (batch_size, 3), the predicted values.
    - batch_y: torch.tensor of shape (batch_size, 3), the true labels.
    - unique_transcription: torch.tensor of shape (n, 3), the tensor containing unique transcriptions.
    - metric_type: str, 'acc' for accuracy or 'mae' for mean absolute error.

    Returns:
    - int or float, the calculated metric (number of correct predictions for accuracy, or total MAE).
    """
    correct = 0
    total_loss = 0.0

    superscripts = '¹²³⁴⁵⁶⁷⁸⁹⁰˥˧˩'

    def remove_superscripts_from_string(text):
        return ''.join(c for c in text if c not in superscripts)

    for index in range(batch_y.size(0)):
        signal = batch_y[index]
        value = values[index]
        pred_seq = match_transcription(value, unique_transcription)

        pred_str = decode_transcription(pred_seq)
        true_str = decode_transcription(signal)

        pred_str_clean = remove_superscripts_from_string(pred_str)
        true_str_clean = true_str  # Ground truth already has no superscripts

        if metric_type == 'acc':
            if pred_str_clean == true_str_clean:
                correct += 1
        elif metric_type == 'mae':
            total_loss += levenshtein_distance(pred_str_clean, true_str_clean)  # or absolute differences

    if metric_type == 'acc':
        return correct
    elif metric_type == 'mae':
        return total_loss

def match_transcription(value, unique_transcription):
    """
    Match the predicted output to the closest valid transcription.

    Args:
    - value: torch.Tensor, the predicted vector
    - unique_transcription: torch.Tensor, list of possible valid vectors

    Returns:
    - closest vector from unique_transcription
    """
    # Compute L2 distance to every unique transcription
    distances = torch.norm(unique_transcription - value, dim=1)
    closest_idx = torch.argmin(distances)
    return unique_transcription[closest_idx]

def compute_cer_wer(reference, hypothesis):
    cer_score = cer(reference, hypothesis)
    wer_score = wer(reference, hypothesis)
    return cer_score, wer_score

def perform_kmeans_clustering(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=kmeans.labels_)
    plt.title('K-Means Clustering of Tone Embeddings')
    plt.show()

def process_files(audio_dir, align_dir, model_name="facebook/wav2vec2-large-xlsr-53"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Tone2Vec(device=device)

    audio_dir = Path(audio_dir)
    align_dir = Path(align_dir)

    all_embeddings = []
    all_transcripts = []

    for wav_file in audio_dir.glob("*.wav"):
        base_name = wav_file.stem
        textgrid_file = align_dir / f"{base_name}.TextGrid"

        if not textgrid_file.exists():
            print(f"Skipping {base_name} (no alignment file found)")
            continue

        waveform, sr = torchaudio.load(wav_file)
        waveform = waveform.squeeze()

        tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=False)
        tier_names = tg.tierNameList
        text_tier_name = tier_names[0]
        segment_tier = tg.tierDict[text_tier_name]
        intervals = segment_tier.entryList

        for idx, (start, end, label) in enumerate(intervals):
            if not label.strip():
                continue

            start_frame = int(start * sr)
            end_frame = int(end * sr)
            segment = waveform[start_frame:end_frame]

            if len(segment) == 0:
                continue

            transcription, tone_embedding = transcribe_audio_segment_with_superscripts(segment.numpy(), sr, model, device)

            all_embeddings.append(tone_embedding)
            all_transcripts.append(transcription)

    all_embeddings = np.vstack(all_embeddings)
    X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_transcripts, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    torch.save(model.state_dict(), "models/tonelab_model.pt")

    y_pred = model.predict(X_test)
    cer_score, wer_score = compute_cer_wer(y_test, y_pred)
    print(f"CER: {cer_score}, WER: {wer_score}")

    perform_kmeans_clustering(all_embeddings)

def transcribe_audio_segment_with_superscripts(segment_audio, sample_rate, model, device):
    tone_representation = model.encode(segment_audio)
    transcription = model.decode(tone_representation)
    
    transcription_with_superscripts = transcription.replace("syllable", "syllable\u00B2") # check to see if this is the correct superscript handling
    return transcription_with_superscripts, tone_representation

def decode_transcription(tensor, idx2char):
    """
    Convert a tensor of indices into a readable string transcription.

    Args:
    - tensor: torch.Tensor of shape (seq_len,) or (batch, seq_len)
    - idx2char: dict mapping index to character/phoneme

    Returns:
    - string transcription
    """
    if tensor.dim() > 1:
        tensor = tensor.squeeze(0)  # remove batch dim if needed

    chars = [idx2char[idx.item()] for idx in tensor if idx.item() in idx2char]
    return ''.join(chars)

def plot_pca(features, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.5)
    
    plt.legend()
    plt.title('PCA of Tone Features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.show()

def plot_tone_clusters(features, labels=None, method='PCA', n_neighbors=15, min_dist=0.1, title='Tone Clusters', save_path=None):
    """
    Visualize tone embeddings in 2D using PCA or UMAP.

    Args:
    - features: np.array or torch.Tensor, shape (n_samples, n_features)
    - labels: Optional list or array of labels for coloring
    - method: 'PCA' or 'UMAP'
    - n_neighbors: UMAP param (ignored for PCA)
    - min_dist: UMAP param (ignored for PCA)
    """

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    if method == 'UMAP':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        proj = reducer.fit_transform(features)
    elif method == 'PCA':
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(features)
    else:
        raise ValueError("method must be 'PCA' or 'UMAP'")

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, cmap='Spectral', s=20, alpha=0.8)

    if labels is not None:
        plt.colorbar(scatter, label='Labels')

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved cluster plot to {save_path}")
    plt.show()


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

"""

