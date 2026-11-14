# DSA Problem Recommender

A web app that recommends LeetCode problems to practise next, based on the problems you have already solved. It finds the topics you have covered least and ranks unsolved problems that are both similar to your solved ones and relevant to those gaps.

**Live demo:** https://dsa-problem-recommender-10.streamlit.app

## Features

- **Pick your solved problems** from the full problem list.
- **Dashboard** with your total solved count and the Easy / Medium / Hard split.
- **Topic-wise progress chart** showing which topics you have practised.
- **Topics to Improve table** comparing how much of your solved set each topic makes up against how common that topic is across all problems in the dataset.
- **Recommendations** (1 to 10) with an optional difficulty filter.
- **Explanation for every recommendation:** either the weak topics it targets or the solved problem it is most similar to.
- **Direct link** to each problem on LeetCode.

## How it works

1. **Data:** a CSV of 3,539 LeetCode problems with title, difficulty, topic tags and URL. Each problem has one or more of 71 topic tags (Array, Hash Table, Dynamic Programming, and so on).
2. **Features:** each problem is represented only by its topic tags. Titles are not used, because title words such as "list" or "sum" made unrelated problems look similar.
3. **Similarity:** topic tags are turned into TF-IDF vectors, so rare topics count for more than very common ones. Cosine similarity then measures how close each unsolved problem is to the problems you solved. A problem's similarity score is its closest match among your solved problems.
4. **Coverage gap:** for each topic, the app computes *(share of all problems with that topic) minus (share of your solved problems with that topic)*, keeping only positive gaps. Non-DSA tags (for example Interactive, Database, Shell) and very rare topics (under 2% of problems) are excluded, so the weak-topic list stays meaningful.
5. **Ranking:** each unsolved problem gets a score of `0.6 × coverage-gap score + 0.4 × similarity`. Problems you have already solved are removed, and the difficulty filter is applied before the top results are returned.

## Tech stack

Python, Pandas, NumPy, scikit-learn (TF-IDF, cosine similarity, MultiLabelBinarizer), Streamlit.

## Run locally

```bash
git clone https://github.com/Dishajindal10/dsa-problem-recommender.git
cd dsa-problem-recommender
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  # Streamlit app: data processing, scoring, UI
leetcode_problems.csv   # Problem dataset
requirements.txt        # Dependencies
README.md
```

## Limitations

- It is a content-based recommender: it only uses topic tags and has no data on what other users solved, so it cannot learn from other people's behaviour.
- The 0.6 / 0.4 weights were chosen by hand. They have not been tuned or evaluated against real user feedback.
- Recommendation quality has not been measured quantitatively.
- Topic tags are a coarse signal. Two problems with the same tags can differ a lot in difficulty and technique.
- The problem list is a static snapshot and does not update with new LeetCode problems.

## Future improvements

- Evaluate the ranking with a hold-out test on simulated users, and tune the weights using it.
- Separate the recommendation logic from the UI and expose it through a REST API (FastAPI).
- Import solved problems by pasting a list or uploading a CSV instead of selecting them one by one.
- Use problem descriptions (text embeddings) in addition to topic tags.

## Author

Disha Jindal: [GitHub](https://github.com/Dishajindal10)
