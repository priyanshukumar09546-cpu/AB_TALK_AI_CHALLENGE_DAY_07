# 🎬 Day 7: Sentiment Analysis Classifier (Klymb AI Challenge)

A machine learning classification model built using Python and `scikit-learn` to classify movie reviews as **Positive** or **Negative**. This project marks the completion of **Sprint 1 (Days 1–7)** of the Klymb AI Challenge.

---

## 📌 Overview
- **Goal:** Build and evaluate a baseline text classification pipeline using NLP preprocessing and supervised learning.
- **Vectorization:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency)
- **Algorithm:** `MultinomialNB` (Multinomial Naive Bayes)
- **Evaluation Metric:** Accuracy Score + 5 Custom Manual Stress-Test Cases

---

## 🛠️ Tech Stack
- **Python 3**
- **Pandas** (Data structuring)
- **Scikit-Learn** (`TfidfVectorizer`, `train_test_split`, `MultinomialNB`, `accuracy_score`)

---

## 🧪 Manual Test Cases & Results
The model was tested on 5 unseen sentences covering clear sentiments, neutral phrasing, negation, and sarcasm:

| Input Review | Expected | Model Prediction | Status |
| :--- | :--- | :--- | :--- |
| *"The visuals were stunning and I really enjoyed it."* | Positive | Negative | ❌ Failed (Unseen words) |
| *"Completely boring and poorly written script."* | Negative | Negative | ✅ Passed |
| *"It was an okay movie, nothing special."* | Neutral/Neg | Positive | ❌ Failed (Subtle phrasing) |
| *"The acting was not bad at all."* | Positive | Positive | ✅ Passed |
| *"Oh great, another two hours of my life wasted."* | Negative | Positive | ❌ Failed (Sarcasm) |

---

## 🔍 Model Reflection: Where Does It Fail & Why?
1. **Small Training Vocabulary:** Trained on a lightweight 10-review sample dataset, unseen positive words like *"stunning"* or *"enjoyed"* were out-of-vocabulary, leading to misclassifications.
2. **Sarcasm & Context:** In *"Oh great, another two hours of my life wasted"*, the classifier latched onto the positive keyword *"great"* and missed the sarcastic intent.
3. **Bag-of-Words Limitation:** TF-IDF + Naive Bayes evaluates word frequencies independently without capturing word order or deeper semantic context.

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
