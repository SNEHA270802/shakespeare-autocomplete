import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

# ✅ Download Shakespeare dataset from TensorFlow
path_to_file = tf.keras.utils.get_file("shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

# Read and clean text
with open(path_to_file, 'r') as f:
    text = f.read().lower()

text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
words = list(set(text.split()))      # unique words

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(words)

# Save vocabulary and model
with open("vocab_list.pkl", "wb") as f:
    pickle.dump(words, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF model and vocabulary saved.")

