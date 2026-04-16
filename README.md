# Sentiment Analysis of IPL 2024 Tweets

## (1) Problem Statement

Social media platforms like Twitter (X) generate millions of opinions daily about major sporting events. During IPL 2024, fans expressed a wide range of emotions — from excitement over match results to frustration with team performance, scheduling, and rule changes. Manually reading and categorising this volume of text is impractical. The problem is to automatically classify the sentiment expressed in IPL 2024 tweets into three categories: **positive**, **neutral**, and **negative**, using machine learning classifiers trained on manually labelled data.

## (2) Objective

- Collect 100 real-world tweets related to IPL 2024 and manually label each as positive, neutral, or negative.
- Build a text classification pipeline using TF-IDF feature extraction and three ML classifiers: Naïve Bayes, SVM, and Logistic Regression.
- Evaluate each model using Precision, Recall, and F1-Score on an unseen test set.
- Compare model performance and identify the most effective classifier for this task.
- Produce visualisations (label distribution, confusion matrices, feature importance) to support findings.

## (3) Dataset

- **Source:** Manually collected from X (formerly Twitter) using search terms `IPL 2024`, `#IPL2024`, and team-specific hashtags (e.g. `#KKR`, `#RCB`, `#CSK`).
- **Features:**
  - `id` — serial number
  - `tweet_text` — raw tweet content
  - `label` — sentiment class: `positive` / `neutral` / `negative`
- **Size:** 100 tweets total — 80 training + 20 test (stratified split)
- **Class distribution:** Positive — 32 | Neutral — 32 | Negative — 36

| File | Rows | Description |
|---|---|---|
| `data/tweets_raw.csv` | 100 | Full labelled dataset |
| `data/train.csv` | 80 | Training split (stratified) |
| `data/test.csv` | 20 | Test split (stratified) |

## (4) Methodology

1. **Data Preprocessing**
   - Tweets collected manually and saved to CSV.
   - Stratified 80/20 train-test split using `sklearn.model_selection.train_test_split`.
   - TF-IDF vectorisation: `max_features=500`, `ngram_range=(1,2)`, `sublinear_tf=True`, `stop_words='english'`.
   - No additional text cleaning applied (URLs and mentions retained as features).

2. **EDA**
   - Label distribution visualised across the full dataset.
   - Tweet character length analysed per sentiment class.
   - Top TF-IDF features per class examined via Logistic Regression coefficients.

3. **Model Building**
   - Three classifiers trained on the TF-IDF feature matrix:
     - `MultinomialNB(alpha=0.5)` — Naïve Bayes
     - `LinearSVC(C=1.0)` — Support Vector Machine
     - `LogisticRegression(C=1.0, max_iter=500)` — Logistic Regression
   - All models trained on the 80-tweet training set.

4. **Evaluation**
   - Predictions made on the 20-tweet test set.
   - Weighted Precision, Recall, and F1-Score computed using `sklearn.metrics`.
   - Confusion matrix generated for each model.
   - Per-class breakdown via `classification_report`.

## (5) Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Naïve Bayes | 0.6678 | 0.6500 | 0.6523 |
| SVM (LinearSVC) | 0.6678 | 0.6500 | 0.6523 |
| Logistic Regression | 0.6588 | 0.6500 | 0.6515 |

**Key Insights:**
- Naïve Bayes and SVM tied for best performance with F1 = **0.6523**.
- All three models produced comparable scores, indicating that on a 100-tweet dataset, TF-IDF feature quality drives performance more than classifier choice.
- Neutral tweets were the hardest class to predict correctly due to vocabulary overlap with both positive and negative tweets.
- Top positive features: `incredible`, `absolute`, `legend`, `never disappoints`.
- Top negative features: `collapse`, `poor`, `boring`, `impact player rule`.

**Visualisations saved in `results/`:**

| File | Description |
|---|---|
| `01_label_distribution.png` | Bar chart of label counts in training set |
| `02_tweet_length_dist.png` | Histogram of tweet lengths by sentiment |
| `03_confusion_matrices.png` | Confusion matrices for all 3 models |
| `04_model_comparison.png` | Grouped bar: Precision / Recall / F1 |
| `05_top_features.png` | Top 10 TF-IDF features per class |

## (6) How to Run

```bash
pip install -r requirements.txt
python main.py
```

Or open the notebook directly:

```bash
jupyter notebook sentiment_analysis.ipynb
```

**Requirements (`requirements.txt`):**
```
pandas
scikit-learn
matplotlib
seaborn
```

**Repository Structure:**
```
├── README.md
├── sentiment_analysis.ipynb
├── main.py
├── requirements.txt
├── data/
│   ├── tweets_raw.csv
│   ├── train.csv
│   └── test.csv
├── results/
│   ├── 01_label_distribution.png
│   ├── 02_tweet_length_dist.png
│   ├── 03_confusion_matrices.png
│   ├── 04_model_comparison.png
│   └── 05_top_features.png
└── reports/
    └── report.pdf
```

## (7) Conclusion

This assignment successfully demonstrated a complete text analytics pipeline for sentiment classification of IPL 2024 tweets. Three classical ML models — Naïve Bayes, SVM, and Logistic Regression — were trained and evaluated on a manually labelled 100-tweet dataset. All three achieved a weighted F1-Score of approximately **0.65**, with Naïve Bayes and SVM performing best. The results confirm that TF-IDF with bigrams is an effective baseline for short-text sentiment classification. Future improvements could include increasing the dataset size to 500+ tweets, applying pre-trained embeddings (BERT or Word2Vec), incorporating emoji and hashtag features, and using cross-validation for more robust evaluation.

## (8) student's details
- Name:Ayush Ramesh Sawant
- Roll No:51
- UIN:231A009
- YEAR: TE-AIDS
