import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('leetcode_problems.csv')
df = df.dropna(subset=['topics'])
df = df.reset_index(drop=True)
df['topics'] = df['topics'].apply(ast.literal_eval)
df['topics_text'] = df['topics'].apply(lambda x: ' '.join(x))
df['combined_text'] = df['title'] + ' ' + df['topics_text']

vectorizer = TfidfVectorizer()
Tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
similarity_matrix = cosine_similarity(Tfidf_matrix)

def get_recommendations(title, top_n = 5, difficulty_filter=None):
    if title not in df['title'].values:
        return None
    
    idx = df[df['title'] == title].index[0]
    
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key = lambda x: x[1], reverse=True)

    top_results = scores[1:top_n+1]

    recommendations = []

    for i, score in scores[1:]:
        problem_difficulty = df.iloc[i]['difficulty']
        if difficulty_filter and problem_difficulty not in difficulty_filter:
            continue 
        recommendations.append({"title": df.iloc[i]['title'],"difficulty": df.iloc[i]['difficulty'],"score": round(float(score),3)})
        if len(recommendations) >= top_n:
            break
    
    return recommendations


# UI:

st.title("DSA Problem Recommender")
st.write("Enter a Leetcode problem you've solved, and get similar problems recommended! ")

problem_titles = df['title'].tolist()
selected_title = st.selectbox("Choose a problem: ", problem_titles)

top_n = st.slider("How many recommendations?", 1, 10, 5) #(min, max, deafult val)

st.write("Filter by difficulty:")
col1, col2, col3 = st.columns(3)
with col1:
    show_easy = st.checkbox("Easy", value=True)
with col2:
    show_medium = st.checkbox("Medium", value=True)
with col3:
    show_hard = st.checkbox("Hard", value=True)

selected_difficulties = []
if show_easy:
    selected_difficulties.append("Easy")
if show_medium:
    selected_difficulties.append("Medium")
if show_hard:
    selected_difficulties.append("Hard")

if(st.button("Get Recommendations")):
    results = get_recommendations(selected_title, top_n, selected_difficulties)
    
    if results:
        st.subheader(f"Problems similar to '{selected_title}' : ")
        for r in results:
            st.write(f"**{r['title']}** - {r['difficulty']} (similarity: {r['score']})")
    else:
        st.write("No recommendations found.")
