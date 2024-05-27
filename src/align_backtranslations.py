from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# File paths
original_file = 'subtask_3_mono_MSA.txt'
translations_file = 'gemini_egyptian_backtranslated.txt'

# Read the text files
original_sentences = read_text_file(original_file)
translation_sentences = read_text_file(translations_file)

# Load the pre-trained sentence embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# Generate embeddings for the original sentences
original_embeddings = model.encode(original_sentences)

# Generate embeddings for the translation sentences
translation_embeddings = model.encode(translation_sentences)

# Find the closest translation sentence and cosine similarity for each original sentence
similarities = cosine_similarity(original_embeddings, translation_embeddings)
closest_indices = similarities.argmax(axis=1)
closest_translations = [translation_sentences[idx] for idx in closest_indices]
closest_similarities = similarities[range(len(original_sentences)), closest_indices]

# Create a DataFrame with the original sentences, closest translations, and cosine similarities
import pandas as pd
df = pd.DataFrame({
    'original': original_sentences,
    'translation': closest_translations,
    'cosine_similarity': closest_similarities
})

# Sort the DataFrame by cosine similarity in descending order and select the top 10,000 rows
top_10k_df = df.nlargest(10000, 'cosine_similarity')

# Save the top 10,000 rows to a CSV file
top_10k_df.to_csv('top_10k_aligned_sentences.csv', index=False)
