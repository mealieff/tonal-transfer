import os
import json
import numpy as np
import librosa
import tgt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def extract_f0(y, sr, point_time):
    idx = int(point_time * sr)
    window = y[max(0, idx - 256):idx + 256]
    pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
    f0 = np.max(pitches, axis=0)
    return np.mean(f0[np.nonzero(f0)]) if np.any(f0) else 0.0


def extract_pitch_contour(y, sr, start_time, end_time, n_points=10):
    """Extracts pitch contour at evenly spaced intervals."""
    times = np.linspace(start_time, end_time, n_points)
    contour = [extract_f0(y, sr, t) for t in times]
    return contour


def extract_mfcc(y, sr, start_time, end_time):
    y_word = y[int(start_time * sr):int(end_time * sr)]
    
    if len(y_word) < 512:
        pad_len = 512 - len(y_word)
        y_word = np.pad(y_word, (0, pad_len), mode='constant')
    n_fft = 512
    hop_length = 128

    mfcc = librosa.feature.mfcc(y=y_word, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    if mfcc.shape[1] >= 5:
        delta_mfcc = librosa.feature.delta(mfcc, width=5)
    else:
        delta_mfcc = np.zeros_like(mfcc)
    return mfcc.mean(axis=1), delta_mfcc.mean(axis=1)


def normalize_f0(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def get_phoneme_context(word_start, word_end, phoneme_tier):
    return [
        intv.text for intv in phoneme_tier
        if intv.start_time >= word_start and intv.end_time <= word_end
    ]

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

        words_tier = tg.get_tier_by_name("Word")
        phoneme_tier = tg.get_tier_by_name("Letter")

        y, sr = librosa.load(audio_path)

        for word_interval in words_tier:
            word = word_interval.text.strip()
            start_time = word_interval.start_time
            end_time = word_interval.end_time

            duration = end_time - start_time
            if duration == 0:
                continue

            mfcc_features, delta_mfcc_features = extract_mfcc(y, sr, start_time, end_time)
            pitch_contour = extract_pitch_contour(y, sr, start_time, end_time)

            phoneme_context = get_phoneme_context(start_time, end_time, phoneme_tier)

            word_data = {
                "word": word,
                "timestamp": {"start": start_time, "end": end_time},
                "duration": duration,
                "pitch_contour": pitch_contour,
                "MFCC_features": mfcc_features.tolist(),
                "Delta_MFCC_features": delta_mfcc_features.tolist(),
                "phoneme_context": phoneme_context
            }

            output_data.append(word_data)

    return output_data


def prepare_data_for_clustering(output_data):
    pitch_contours = []
    mfcc_features = []
    delta_mfcc_features = []
    
    for word_data in output_data:
        pitch_contours.append(word_data["pitch_contour"])
        mfcc_features.append(word_data["MFCC_features"])
        delta_mfcc_features.append(word_data["Delta_MFCC_features"])

    pitch_contours = np.array(pitch_contours)
    mfcc_features = np.array(mfcc_features)
    delta_mfcc_features = np.array(delta_mfcc_features)

    pitch_contours = normalize_f0(pitch_contours)
    feature_data = np.hstack([pitch_contours, mfcc_features, delta_mfcc_features])

    return feature_data

def perform_kmeans_clustering(features, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    silhouette = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette}")
    
    return labels, kmeans


def plot_tonal_contours_with_clusters(vectors, labels):
    plt.figure(figsize=(10, 6))
    n_clusters = np.max(labels) + 1
    colors = sns.color_palette("Set1", n_clusters)

    for cluster_id in range(n_clusters):
        cluster_vectors = vectors[labels == cluster_id]
        for vec in cluster_vectors:
            plt.plot(vec, color=colors[cluster_id], alpha=0.6)
    
    plt.title("Tonal Contours Grouped by Clusters")
    plt.xlabel("Time (Normalized)")
    plt.ylabel("Pitch (F0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("updated_tone_clusters.png")
    print("Saved plot to tone_clusters.png")

def plot_chao_tones(vectors, labels):
    n_clusters = np.max(labels) + 1
    for cluster_id in range(n_clusters):
        cluster_vectors = vectors[labels == cluster_id]
        plt.figure(figsize=(8, 6))
        for vec in cluster_vectors:
            chao = np.interp(vec, (min(vec), max(vec)), (1, 5))  # Normalize to Chao scale (1-5)
            plt.plot([0, 1, 2], chao, marker='o', alpha=0.6)
        plt.title(f"Cluster {cluster_id} - Tonal Contours (Chao Scale)")
        plt.xticks([0, 1, 2], ['Initial', 'Mid', 'Final'])
        plt.yticks([1, 2, 3, 4, 5])
        plt.ylim(1, 5)
        plt.xlabel("Position")
        plt.ylabel("Tone Height (Chao Scale)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"updated_tone_clusters_chao_{cluster_id}.png")
        print(f"Saved plot to tone_clusters_chao_{cluster_id}.png")

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def save_to_json(data, out_path):
    serializable_data = convert_to_serializable(data)
    with open(out_path, "w") as f:
        json.dump(serializable_data, f, indent=2)

def main(textgrid_folder, audio_folder, n_clusters=4):
    output_data = process_textgrid(textgrid_folder, audio_folder)
    features = prepare_data_for_clustering(output_data)
    save_to_json(output_data, "lamkang_word_extendedfeatures.json")
    labels, kmeans = perform_kmeans_clustering(features, n_clusters)

    plot_tonal_contours_with_clusters(features, labels)
    plot_chao_tones(features, labels)


if __name__ == "__main__":
    textgrid_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts/Forced_Aligned"
    audio_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts"
    main(textgrid_folder, audio_folder, n_clusters=4)

