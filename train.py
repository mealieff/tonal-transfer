import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from jiwer import cer, wer
import tonelab.tone2vec
import importlib
importlib.reload(tonelab.tone2vec)
from tonelab.tone2vec import loading, parse_phonemes, tone_feats, plot
from tonelab.model import VGG, ResNet, DenseNet, mlp
import json
from parse_eaf.parse_eaf import process_eaf_data, create_vocab

def load_data(audio_dir, align_dir):
    dataset_path = audio_dir  
    info_path = align_dir    

    dataset = loading(dataset_path)
    labels = loading(info_path, column_name='areas')

    _, _, _, tone_list = parse_phonemes(dataset)
    feats = tone_feats(tone_list)

    X = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels.values, dtype=torch.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    unique_transcription = X.unique(dim=0)  

    return train_loader, valid_loader, unique_transcription

def preprocess_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    eaf_files = [f for f in os.listdir(data_dir) if f.endswith(".eaf")]

    train_data = []
    for eaf_file in eaf_files:
        wav_file = eaf_file.replace(".eaf", ".wav")
        wav_path = os.path.join(data_dir, "audio", wav_file)
        eaf_path = os.path.join(data_dir, eaf_file)
        if os.path.exists(wav_path):
            train_data.extend(process_eaf_data(eaf_path, wav_path))

    with open(os.path.join(output_dir, "train_segments.jsonl"), "w", encoding="utf-8") as f:
        for fn in dataset:
            text = open(os.path.join(data_dir, fn)).read()
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    vocab = create_vocab(text)
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


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

class mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def main():
    parser = argparse.ArgumentParser(description="Run Tone2Vec processing and evaluation on Lamkang data.")
    parser.add_argument("--do_preprocess", action="store_true", help="Run preprocessing before training.")
    parser.add_argument('--audio_dir', type=str, default='/lamkang_data/Lamkang_aligned_audio_and_transcripts', help='Directory with .wav files')
    parser.add_argument('--align_dir', type=str, default='/lamkang_data/Lamkang_aligned_audio_and_transcript/Forced_Aligned', help='Directory with .TextGrid alignment files')
    parser.add_argument('--save_model', type=str, default='models/tonelab_model.pt', help='Where to save the trained model')
    parser.add_argument('--do_train', action='store_true', help='If set, will train a classifier on extracted embeddings')
    parser.add_argument('--do_eval', action='store_true', help='If set, will evaluate WER and CER')
    parser.add_argument('--do_cluster', action='store_true', help='If set, will visualize tone clusters using KMeans')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--print_interval', type=int, default=5, help='How often to print training results')
    parser.add_argument("--output_dir", type=str, default="~/output", help='tricky bc actually output of preprocessing, also known as input')

    args = parser.parse_args()
    if args.do_preprocess:
        preprocess_data(args.align_dir, args.output_dir)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        'VGG': VGG(),
        'ResNet': ResNet(),
        'DenseNet': DenseNet(),
        'MLP': mlp(input_size=128, hidden_sizes=[64, 32]),
         }

    best_accuracy = 0
    best_model_name = ''
    results = {}

    train_loader, valid_loader, unique_transcription = load_data(args.audio_dir, args.align_dir)

    for model_name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        accuracy = train_and_evaluate(model, train_loader, valid_loader, optimizer, device, args.num_epochs, args.save_model, unique_transcription, args.print_interval)
        results[model_name] = accuracy
        print(f"Model: {model_name}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name

    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

    if args.do_cluster:
        # Perform clustering visualization
        print("Performing KMeans clustering...")
        embeddings, labels = extract_embeddings_for_clustering(train_loader)
        perform_kmeans_clustering(embeddings)

    if args.do_eval:
        # Perform evaluation (WER, CER, etc.)
        print("Evaluating performance...")
        # Add evaluation logic here as needed

if __name__ == '__main__':
    main()

"""
Example usage:
first time: 
python3  --do_preprocess  --do_cluster

python3 --do_cluster
python3 --do_train --do_eval


"""

