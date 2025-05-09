import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def normalize_zscore(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std if std > 0 else arr - mean

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_tone_vectors(data):
    f0_vectors = []
    words = []
    mfcc_vectors = []
    for item in data:
        tone_vec = [item["initial_f0"], item["mid_f0"], item["final_f0"]]
        if any(f == 0.0 for f in tone_vec):
            continue
        norm_f0 = normalize_zscore(tone_vec)
        f0_vectors.append(norm_f0)
        words.append(item["word"])
        mfcc = item.get("mfcc", [])
        mfcc_vectors.append(mfcc)
    return np.array(f0_vectors), words, np.array(mfcc_vectors)

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


def plot_chao_tones(f0_vectors, labels):
    def chao_scale(f0_values):
        return np.interp(np.log2(f0_values), [5.0, 9.0], [1, 5])
    
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
    f0_vectors, words, mfcc_vectors = extract_tone_vectors(data)

    if len(f0_vectors) < n_clusters:
        print("Not enough data for the requested number of clusters.")
        return

    # F0-only clustering
    kmeans_f0 = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels_f0 = kmeans_f0.fit_predict(f0_vectors)
    sil_f0 = silhouette_score(f0_vectors, labels_f0)
    print(f"[F0-only] Silhouette Score: {sil_f0:.3f}")
    print(f"[F0-only] Cluster Counts: {np.bincount(labels_f0)}")
    plot_clusters(f0_vectors, labels_f0, "F0-Only Clusters", "tone_clusters_f0.png")
    plot_chao_tones(f0_vectors, labels_f0)

    # Save output
    for item, label in zip(data, labels_f0):
        item["cluster_id_f0"] = int(label)
    with open("lamkang_clustered_f0.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved F0-only cluster results to lamkang_clustered_f0.json")

    # F0 + MFCC clustering if MFCCs are non-empty
    if mfcc_vectors.size > 0 and len(mfcc_vectors[0]) > 0:
        combined = np.hstack([f0_vectors, mfcc_vectors])
        kmeans_multi = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels_multi = kmeans_multi.fit_predict(combined)
        sil_multi = silhouette_score(combined, labels_multi)
        print(f"[F0+MFCC] Silhouette Score: {sil_multi:.3f}")
        print(f"[F0+MFCC] Cluster Counts: {np.bincount(labels_multi)}")
        plot_clusters(combined, labels_multi, "F0 + MFCC Clusters", "tone_clusters_multivariate.png")

        for item, label in zip(data, labels_multi):
            item["cluster_id_multivariate"] = int(label)
        with open("lamkang_clustered_multivariate.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved multivariate cluster results to lamkang_clustered_multivariate.json")

if __name__ == "__main__":
    main("lamkang_word_features.json", n_clusters=4)
