import os
import librosa
import numpy as np
import tonelab.tone2vec
import importlib
importlib.reload(tonelab.tone2vec)
from tonelab.tone2vec import loading, parse_phonemes, tone_feats, plot
from tonelab.model import VGG, ResNet, DenseNet, mlp
from tonelab.clustering import auto_cluster_feat, auto_cluster_plot
from sklearn.decomposition import PCA

def tone_feats(audio, sr, weights_path='ToneLab/tonelab/weights/tone2vec.npy'):
    data_tone2vec = np.load(weights_path)
    reshaped_data = data_tone2vec.reshape(-1, 216)  # Assuming data_tone2vec is originally (6, 6, 6, 6, 6, 6)

    model = PCA(n_components=2) 
    reduced_data = model.fit_transform(reshaped_data)
    
    print(f"Shape of reduced data: {reduced_data.shape}")  # Debugging line
    return reduced_data

wav_dir = "lamkang_data/Lamkang_aligned_audio_and_transcripts/"

wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
tone_vectors = []

for wav_file in wav_files:
    wav_path = os.path.join(wav_dir, wav_file)
    audio, sr = librosa.load(wav_path, sr=None)  # Load with original sampling rate
    tone_vector = tone_feats(audio, sr, weights_path='ToneLab/tonelab/weights/tone2vec.npy')

    if tone_vector is not None and not np.isnan(tone_vector).any():
        tone_vectors.append(tone_vector)
    else:
        print(f"Skipping {wav_file} because tone vector is None or contains NaNs")

tone_vectors_flat = np.array(tone_vectors).reshape(len(tone_vectors), -1)
print(f"Flattened tone_vectors shape: {tone_vectors_flat.shape}")
tone_vectors_flat = tone_vectors_flat[~np.isnan(tone_vectors_flat).any(axis=1)]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
tone_vectors_2d = pca.fit_transform(tone_vectors_flat)

print("Explained variance ratio for each component: ", pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
plt.scatter(tone_vectors_2d[:, 0], tone_vectors_2d[:, 1])
plt.title("Tonal Embedding of Lamkang Tones")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

