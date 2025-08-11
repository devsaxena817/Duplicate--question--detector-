# 🧠 Duplicate Question Detector

A machine learning project that detects whether two given questions are duplicates or not — useful for Q&A forums like Quora or Stack Overflow.

---
## 🚀 Live Demo
Try the app here: [Duplicate Question Detector](https://2rzqgbcteq4hhxysadb5gf.streamlit.app/)

## 📌 Features
- **BERT-based sentence embeddings** (via SentenceTransformers `all-MiniLM-L6-v2`)
- **Custom NLP features**: common words count, length-based features, fuzzy matching
- **XGBoost model** for high accuracy classification
- Streamlit web app for easy interaction

---

## 🚀 How It Works
1. **Data Processing**  
   - Preprocess text (lowercasing, stopword removal, punctuation removal)
   - Extract NLP-based similarity features
   - Generate sentence embeddings using SBERT

2. **Model Training**  
   - Combine handcrafted features + embeddings into a single feature vector  
   - Train with XGBoost and fine-tune parameters using GridSearch

3. **Prediction**  
   - User enters two questions in the Streamlit app  
   - Model predicts whether they are duplicates

---

## 🖥️ Running Locally

### 1️⃣ Clone the repository
```
git clone https://github.com/devsaxena817/Duplicate--question--detector-.git
cd Duplicate--question--detector-
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Run the Web App
```
streamlit run app.py
```
---

## 📊 Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Classifier**: XGBoost (GPU enabled)
- **Feature Count**: ~791 per question pair
- **Dataset Size**: ~30,000 rows
- **Best Parameters**: Tuned with GridSearchCV

---

## 🛠 Requirements

See [`requirements.txt`](requirements.txt) for the full list, including:
- `sentence-transformers`
- `scikit-learn`
- `xgboost`
- `nltk`
- `streamlit`
- `numpy`
- `pandas`

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Author

**Dev Saxena**  
[GitHub](https://github.com/devsaxena817)  


