# Sentiment Classifier
A machine learning project that classifies movie reviews as **positive** or **negative** using the IMDB dataset. It demonstrates the full pipeline from data cleaning to model training, evaluation, and saving using Scikit-learn.

📌 Key Features
- TF-IDF vectorization of text data
- Binary classification with:
  - SVM (Support Vector Machine)
  - Random Forest
- Model evaluation with precision, recall, and F1-score
- Saves trained model and vectorizer for deployment

🧹 Preprocessing Steps
- Lowercasing and removing HTML tags
- Cleaning non-alphabet characters
- Removing extra spaces

📊 Evaluation Metrics
Each model is evaluated using:
- 🟣 Precision
- 🔵 Recall
- 🟢 F1-Score

📦 Files Included
- `sentiment_classifier.ipynb` – Full Jupyter Notebook
- `sentiment_model.pkl` – Trained SVM model
- `tfidf_vectorizer.pkl` – Trained TF-IDF vectorizer

📁 Dataset
**IMDB Dataset of 50K Movie Reviews**  
Download from:  
[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
➡ After download, place `IMDB Dataset.csv` in the project root.

⚙️ Tech Stack
- Python
- Pandas / Numpy
- Scikit-learn
- Jupyter Notebook
- (Optional) Flask for API

🙌 Contributions
Feel free to fork, improve the model, or add support for deep learning using LSTM/BERT!
