import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def vector_similarity(text1, text2):
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])

    # Calculate cosine similarity
    return cosine_similarity(embedding1, embedding2)[0][0]


def paragraph_keywords_similarity(paragraph, keywords):
    # Combine keywords into a single string
    keywords_text = " ".join(keywords)

    # Calculate vector similarity
    similarity_score = vector_similarity(paragraph, keywords_text)

    return similarity_score