import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def chao_scale(f0s):
    """Map normalized F0s to Chao's 1–5 scale safely."""
    f0s = np.array(f0s)
    if np.any(f0s <= 0):
        f0s = f0s + 1e-5  # prevent log2(0)
    f0s_log = np.log2(f0s)
    min_log, max_log = np.min(f0s_log), np.max(f0s_log)
    if max_log - min_log == 0:
        return np.array([3] * len(f0s))  # neutral tone level
    scaled = 1 + 4 * (f0s_log - min_log) / (max_log - min_log)
    return np.round(scaled).astype(int)


def normalize_zscore(vectors):
    vectors = np.array(vectors)
    if len(vectors) == 0:
        print("Warning: empty input to normalize_zscore.")
        return vectors
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    std[std == 0] = 1e-6  # Prevent division by zero
    return (vectors - mean) / std


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_features(data):
    f0_vectors = []
    mfcc_vectors = []
    words = []
    for item in data:
        if any(f == 0.0 for f in [item["initial_f0"], item["mid_f0"], item["final_f0"]]):
            continue
        f0_vec = [item["initial_f0"], item["mid_f0"], item["final_f0"]]
        mfcc_vec = item.get("mfcc", [])
        f0_vectors.append(f0_vec)
        mfcc_vectors.append(mfcc_vec)
        words.append(item["word"])
    return np.array(f0_vectors), np.array(mfcc_vectors), words

def plot_clusters(vectors, labels, title, filename):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")


def plot_chao_tones(f0_vectors, labels, prefix="tone_clusters_chao"):
    n_clusters = np.max(labels) + 1
    for cluster_id in range(n_clusters):
        cluster_vectors = f0_vectors[labels == cluster_id]
        plt.figure()
        for vec in cluster_vectors:
            chao = chao_scale(np.maximum(vec, 1e-3))  # Avoid log(0)
            plt.plot([0, 1, 2], chao, marker='o', alpha=0.6)
        plt.title(f"Cluster {cluster_id} - Tonal Contours")
        plt.xticks([0, 1, 2], ['Initial', 'Mid', 'Final'])
        plt.yticks([1, 2, 3, 4, 5])
        plt.ylim(1, 5)
        plt.xlabel("Position")
        plt.ylabel("Tone Height (Chao Scale)")
        plt.grid(True)
        plt.tight_layout()
        filename = f"tone_clusters_chao_{cluster_id}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot to {filename}")

def main(json_path, n_clusters=4):
    data = load_json(json_path)
    f0_vectors, mfcc_vectors, words = extract_features(data)

    if len(f0_vectors) < n_clusters:
        print("Not enough data for the requested number of clusters.")
        return

    # Normalize
    f0_norm = normalize_zscore(f0_vectors)
    mfcc_norm = normalize_zscore(mfcc_vectors)

    # ----- F0-only clustering -----
    kmeans_f0 = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels_f0 = kmeans_f0.fit_predict(f0_norm)
    sil_f0 = silhouette_score(f0_norm, labels_f0)
    print(f"[F0 Only] Silhouette Score: {sil_f0:.3f}")
    print(f"[F0 Only] Cluster Counts: {np.bincount(labels_f0)}")

    plot_clusters(f0_norm, labels_f0, "F0-Only Clustering", "tone_clusters_f0.png")
    plot_chao_tones(f0_vectors, labels_f0, prefix="tone_clusters_chao_f0")

    # ----- Multivariate (F0 + MFCC) clustering -----
    combined = np.hstack((f0_norm, mfcc_norm))
    kmeans_comb = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels_comb = kmeans_comb.fit_predict(combined)
    sil_comb = silhouette_score(combined, labels_comb)
    print(f"[F0 + MFCC] Silhouette Score: {sil_comb:.3f}")
    print(f"[F0 + MFCC] Cluster Counts: {np.bincount(labels_comb)}")

    plot_clusters(combined, labels_comb, "F0 + MFCC Clustering", "tone_clusters_multivariate.png")
    plot_chao_tones(f0_vectors, labels_comb, prefix="tone_clusters_chao_comb")

    # Save output with both cluster labels
    output = []
    for i, item in enumerate(data):
        if i >= len(labels_f0):
            continue
        if all(f > 0 for f in [item["initial_f0"], item["mid_f0"], item["final_f0"]]):
            item["f0_cluster"] = int(labels_f0[i])
            item["f0_mfcc_cluster"] = int(labels_comb[i])
            output.append(item)

    with open("lamkang_clustered.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Cluster assignments saved to lamkang_clustered.json")

if __name__ == "__main__":
    main("lamkang_word_features.json", n_clusters=4)
