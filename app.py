import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="DSA Problem Recommender",
    page_icon="🎯",
    layout="wide"
)

df = pd.read_csv('leetcode_problems.csv')
df = df.dropna(subset=['topics'])
df = df.reset_index(drop=True)
df['topics'] = df['topics'].apply(ast.literal_eval)
df['topics_text'] = df['topics'].apply(lambda x: ' '.join(x))

EXCLUDE = {"Interactive", "Concurrency", "Database", "Shell", "Brainteaser"}

mlb = MultiLabelBinarizer()
T = mlb.fit_transform(df["topics"])
topic_names = mlb.classes_
dataset_share = T.mean(axis=0)
keep = np.array([
    (t not in EXCLUDE) and (dataset_share[i] >= 0.02)
    for i, t in enumerate(topic_names)
])

X = TfidfVectorizer(analyzer=lambda t: t).fit_transform(df["topics"])


def weak_topic_weights(solved_idx):
    user_share = T[solved_idx].mean(axis=0)
    gap = np.clip(dataset_share - user_share, 0, None) * keep
    return gap / gap.max() if gap.max() > 0 else gap


def get_recommendations(titles, top_n=5, difficulties=None):
    solved = df.index[df["title"].isin(titles)].to_numpy()
    if len(solved) == 0:
        return []

    w = weak_topic_weights(solved)

    sim_all = cosine_similarity(X[solved], X)
    sim = sim_all.max(axis=0)
    src = sim_all.argmax(axis=0)

    gap_score = (T @ w) / np.maximum(T.sum(axis=1), 1)
    score = 0.6 * gap_score + 0.4 * sim

    score[solved] = -1
    if difficulties:
        score[~df["difficulty"].isin(difficulties).to_numpy()] = -1

    out = []
    for i in np.argsort(-score)[:top_n]:
        if score[i] < 0:
            break

        weak_hit = [topic_names[j]
                    for j in np.where((T[i] == 1) & (w > 0.3))[0]]
        out.append({
            "title": df.at[i, "title"],
            "difficulty": df.at[i, "difficulty"],
            "url": df.at[i, "url"],
            "score": round(float(score[i]), 3),
            "weak_topics": weak_hit,
            "similar_to": df.at[solved[src[i]], "title"],
        })
    return out

# ***** UI:


st.title("DSA Problem Recommender")
st.write("Enter a Leetcode problem you've solved, and get similar problems recommended! ")


# select problems:
problem_titles = df['title'].tolist()
selected_title = st.multiselect(
    "Select problems you've already solved: ", problem_titles)

st.header("Your Dashboard")

# problem selected by user:
selected_problems = df[df['title'].isin(selected_title)]

total_solved = len(selected_problems)

difficulty_counts = (
    selected_problems['difficulty']
    .value_counts())

easy_count = difficulty_counts.get("Easy", 0)
medium_count = difficulty_counts.get("Medium", 0)
hard_count = difficulty_counts.get("Hard", 0)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Solved", total_solved)

col2.metric(
    "Easy", easy_count)

col3.metric(
    "Medium", medium_count)

col4.metric(
    "Hard", hard_count)

st.divider()

st.subheader("Topic-wise Progress")

if total_solved > 0:
    topic_counts = (
        selected_problems['topics']
        .explode()
        .value_counts()
    )

    # Display topic-wise chart
    st.bar_chart(
        topic_counts.head(10),
        x_label="Topics",
        y_label="Problems Solved"
    )

    # Identify least-practised topics
    st.subheader("Topics to Improve")

    solved_idx = selected_problems.index.to_numpy()
    w = weak_topic_weights(solved_idx)
    user_share = T[solved_idx].mean(axis=0)

    weak_df = pd.DataFrame({
        "Topic": topic_names,
        "You (%)": (user_share * 100).round(1),
        "Dataset (%)": (dataset_share * 100).round(1),
        "gap": w,
    })
    weak_df = weak_df[weak_df["gap"] > 0].sort_values(
        "gap", ascending=False).head(5)

    st.dataframe(
        weak_df[["Topic", "You (%)", "Dataset (%)"]],
        use_container_width=True,
        hide_index=True
    )


else:
    st.info(
        "Select the problems you have solved "
        "to see your DSA progress."
    )

st.divider()


if selected_title:
    selected_problems = df[df["title"].isin(selected_title)]

    with st.expander(f"View your {len(selected_problems)} solved problems"):
        st.dataframe(
            selected_problems[["title", "difficulty", "topics_text"]],
            use_container_width=True,
            hide_index=True
        )

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
            with st.container(border=True):
                st.subheader(r["title"])

                st.caption(
                    f"{r['difficulty']}  |  Recommendation score: {r['score']}"
                )

                if r["weak_topics"]:
                    st.write("**Why this?** Targets your weak topics: " +
                             ", ".join(r["weak_topics"]))
                else:
                    st.write("**Why this?** Similar to: " + r["similar_to"])

                st.link_button("Solve problem ↗", r["url"])

    else:
        st.write("No recommendations found.")
