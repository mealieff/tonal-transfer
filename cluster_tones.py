import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def chao_scale(f0s):
    """Map normalized F0s to Chao's 1–5 scale."""
    f0s = np.array(f0s)
    if np.all(f0s == 0):
        return [0] * len(f0s)
    min_f0, max_f0 = np.min(f0s), np.max(f0s)
    normalized = 1 + 4 * (f0s - min_f0) / (max_f0 - min_f0)
    return np.round(normalized).astype(int)

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_tone_vectors(data):
    vectors = []
    words = []
    for item in data:
        tone_vec = [item["initial_f0"], item["mid_f0"], item["final_f0"]]
        if any(f == 0.0 for f in tone_vec):  # Skip incomplete contours
            continue
        vectors.append(tone_vec)
        words.append(item["word"])
    return np.array(vectors), words

def plot_clusters(vectors, labels, words):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title("KMeans Clusters of Tonal Contours")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
#    plt.show()
    plt.savefig("tone_clusters.png")
    print("Saved plot to tone_clusters.png")

def plot_chao_tones(vectors, labels):
    n_clusters = np.max(labels) + 1
    for cluster_id in range(n_clusters):
        cluster_vectors = vectors[labels == cluster_id]
        plt.figure()
        for vec in cluster_vectors:
            chao = chao_scale(vec)
            plt.plot([0, 1, 2], chao, marker='o', alpha=0.6)
        plt.title(f"Cluster {cluster_id} - Tonal Contours")
        plt.xticks([0, 1, 2], ['Initial', 'Mid', 'Final'])
        plt.yticks([1, 2, 3, 4, 5])
        plt.ylim(1, 5)
        plt.xlabel("Position")
        plt.ylabel("Tone Height (Chao Scale)")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig("tone_clusters_chao.png")
        print("Saved plot to tone_clusters_chao.png")

def main(json_path, n_clusters=4):
    data = load_json(json_path)
    vectors, words = extract_tone_vectors(data)
    
    if len(vectors) < n_clusters:
        print("Not enough data for the requested number of clusters.")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(vectors)
    
    sil_score = silhouette_score(vectors, labels)
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Cluster Counts: {np.bincount(labels)}")

    plot_clusters(vectors, labels, words)
    plot_chao_tones(vectors, labels)

    output = []
    for item, label in zip(data, labels):
        if all(f > 0 for f in [item["initial_f0"], item["mid_f0"], item["final_f0"]]):
            item["cluster_id"] = int(label)
            output.append(item)
    with open("lamkang_clustered.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Cluster assignments saved to lamkang_clustered.json")

if __name__ == "__main__":
    main("lamkang_word_features.json", n_clusters=4)

