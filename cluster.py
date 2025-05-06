import os
import numpy as np
import librosa
from textgrid import TextGrid
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tonelab.tone2vec import cal_simi, Levenshtein_distance, loading, parse_phonemes, tone_feats, plot

textgrid_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts/Forced_Aligned"
audio_folder = "lamkang_data/Lamkang_aligned_audio_and_transcripts"

def extract_pitch_contours(audio_folder, textgrid_folder):
    pitch_contours = {}
    for file in os.listdir(audio_folder):
        if file.endswith(".wav"):
            audio_file = os.path.join(audio_folder, file)
            tg_file = os.path.join(textgrid_folder, file.replace(".wav", ".TextGrid"))
            y, sr = librosa.load(audio_file)
            pitches, _ = librosa.core.piptrack(y=y, sr=sr)
            pitch_contour = np.mean(pitches, axis=0)  # Average pitch contour

            pitch_contours[file] = pitch_contour
    return pitch_contours

def get_word_alignments(textgrid_folder):
    alignments = {}
    for file in os.listdir(textgrid_folder):
        if file.endswith(".TextGrid"):
            tg = textgrid.TextGrid.fromFile(os.path.join(textgrid_folder, file))
            words_tier = tg[1]

            word_alignments = []
            for interval in words_tier:
                word = interval.mark
                start_time = interval.xmin
                end_time = interval.xmax
                word_alignments.append((word, start_time, end_time))

            alignments[file] = word_alignments
    return alignments

def map_embeddings_to_words(word_alignments, pitch_contours):
    word_embedding_mapping = {}
    for file, words in word_alignments.items():
        pitch_contour = pitch_contours.get(file)

        if pitch_contour is not None:
            initials_lists, finals_lists, all_lists, tones_lists = parse_phonemes(pitch_contour)
            tone_embedding = tone_feats(tones_lists, method='PCA', dim=2)  # Example with PCA
            for word, start, end in words:
                word_embedding_mapping[(file, word)] = tone_embedding

    return word_embedding_mapping

def apply_umap(embeddings, n_components=2):
    umap_model = umap.UMAP(n_components=n_components)
    umap_embeddings = umap_model.fit_transform(embeddings)
    return umap_embeddings

def cluster_and_visualize(embeddings, n_clusters=5, n_components=2):
    umap_embeddings = apply_umap(embeddings, n_components)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=kmeans_labels, cmap='viridis', s=5)
    plt.title(f"UMAP projection of tone embeddings with KMeans clustering (n_clusters={n_clusters})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar()
    plt.show()

def main():
    pitch_contours = extract_pitch_contours(audio_folder, textgrid_folder)
    word_alignments = get_word_alignments(textgrid_folder)
    word_embedding_mapping = map_embeddings_to_words(word_alignments, pitch_contours, tone2vec_model)
    embeddings = np.array(list(word_embedding_mapping.values()))
    cluster_and_visualize(embeddings, n_clusters=5, n_components=2)

if __name__ == "__main__":
    main()

