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


def get_recommendations(titles, top_n=5, difficulty_filter=None):
    solved_indices = df.index[df["title"].isin(titles)].tolist()

    if not solved_indices:
        return []

    combined_scores = []

    for i in range(len(df)):

        # Skip problems the user has already solved
        if i in solved_indices:
            continue

        # Get similarity with each solved problem
        similarities = [
            similarity_matrix[solved_index][i]
            for solved_index in solved_indices
        ]

        score = max(similarities)

        combined_scores.append((i, score))

    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i, score in combined_scores:
        problem_difficulty = df.iloc[i]['difficulty']

        if difficulty_filter and problem_difficulty not in difficulty_filter:
            continue


        candidate_topics = set(df.iloc[i]["topics"])

        solved_topics = set()

        for solved_index in solved_indices:
            solved_topics.update(df.iloc[solved_index]["topics"])

        matched_topics = candidate_topics.intersection(solved_topics)

        recommendations.append({
            "title": df.iloc[i]['title'],
            "difficulty": df.iloc[i]['difficulty'],
            "score": round(float(score), 3),
            "url": df.iloc[i]['url'],
            "matched_topics": list(matched_topics)
        })

        if len(recommendations) >= top_n:
            break

    return recommendations


# ***** UI:

st.title("DSA Problem Recommender")
st.write("Enter a Leetcode problem you've solved, and get similar problems recommended! ")


# select problems:
problem_titles = df['title'].tolist()
selected_title = st.multiselect(
    "Select problems you've already solved: ", problem_titles)

if (selected_title):
    selected_problems = df[df['title'].isin(selected_title)]

    st.write("### Your Selected Problems: ")
    st.dataframe(selected_problems[["title", "difficulty", "topics_text"]])


top_n = st.slider("How many recommendations?", 1, 10, 5)

st.write("Filter by difficulty:")
col1, col2, col3 = st.columns(3)
with col1:
    show_easy = st.checkbox("Easy", value=False)
with col2:
    show_medium = st.checkbox("Medium", value=False)
with col3:
    show_hard = st.checkbox("Hard", value=False)

selected_difficulties = []
if show_easy:
    selected_difficulties.append("Easy")
if show_medium:
    selected_difficulties.append("Medium")
if show_hard:
    selected_difficulties.append("Hard")

if (st.button("Get Recommendations")):
    results = get_recommendations(selected_title, top_n, selected_difficulties)

    if results:
        st.subheader(f"Recommended Problems")

        for r in results:
            st.write(
                f"### {r['title']}"
            )

            st.write(
                f"**Difficulty:** {r['difficulty']}"
                f" | **Similarity:** {r['score']}"
            )

            if r["matched_topics"]:
                st.write(
                    "**Why recommendation?** " + ", ".join(r["matched_topics"])
                )

            st.link_button("🔗 Solve the problem", r["url"])
    else:
        st.write("No recommendations found.")
