import pandas as pd
import numpy as np
import ast
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load data:
df = pd.read_csv('leetcode_problems.csv')
df = df.dropna(subset=['topics'])
df = df.reset_index(drop = True)
df['topics'] = df['topics'].apply(ast.literal_eval)
df['topics_text'] = df['topics'].apply(lambda x: ' '.join(x))
df['combined_text'] = df['title'] + ' ' + df['topics_text']

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
similarity_matrix = cosine_similarity(tfidf_matrix)

def get_recommendations(title, top_n = 5):
    if title not in df['title'].values:
        return f"'{title}' not found in dataset."

    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_results = scores[1:top_n+1]

    recommendations = []
    for i, score in top_results:
        recommendations.append({
            "title" : df.iloc[i]['title'],
            "difficulty" : df.iloc[i]['difficulty'],
            "score" : round(float(score), 3)
        })
    return recommendations

def precision_at_k (recommended_titles, relevant_titles, k=5):
    recommended_top_k = recommended_titles[:k]
    relevant_cnt = sum(1 for title in recommended_top_k if title in relevant_titles)
    return relevant_cnt / k





@app.get("/")
def read_root():
    return {"message": "Hello, this is my recommender API"}

@app.get("/recommend_problem/{title}")
def recommend_problem(title:str):
    return get_recommendations(title)