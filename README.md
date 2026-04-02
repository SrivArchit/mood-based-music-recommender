# Hybrid Mood-Based Music Recommender 🎶

A sophisticated, multi-layered song recommendation system that suggests Spotify tracks based on a user's **current mood**, **listening history**, and **context**.

This project demonstrates an **end-to-end machine learning pipeline**, from fine-tuning an advanced NLP model (**DistilBERT**) to deploying a hybrid recommendation engine in an **interactive web application** using Streamlit.

---

## ✨ Key Features

- **Advanced Mood Detection**  
  Fine-tunes the **DistilBERT** transformer model for nuanced, multi-label emotion classification from user text using Google's GoEmotions dataset.

- **Hybrid Filtering Engine**  
  Combines three powerful recommendation techniques for highly personalized and relevant suggestions:  
  1. **Collaborative Filtering** – SVD model recommends songs based on listening patterns of similar users.  
  2. **Content-Based Filtering** – Re-ranks recommendations using cosine similarity between a song's audio features (e.g., valence, energy) and the detected mood.  
  3. **Contextual Filtering** – Further refines suggestions based on situational context (e.g., recommending calmer music in the evening).

- **Interactive Web Application**  
  A **Streamlit**-based interface for real-time interaction with the recommender system.

---



## 🛠 Tech Stack

- **Backend & ML:** Python, PyTorch, Transformers, Scikit-learn (F1, AUC, Accuracy, cosine_similarity), Scikit-surprise(Dataset, Reader, SVD), Pandas, NumPy
- **Model Fine-Tuning:** Datasets (Hugging Face), AutoTokenizer, AutoModelForSequenceClassification, Trainer,  EarlyStoppingCallback
- **Frontend:** Streamlit  
- **NLP Tools:** NLTK  



---

## 🚀 Local Setup and Installation

### **Prerequisites**
- Python 3.8+
- `pip` and `venv`

### **0️⃣ Download dataset and fine-tune the model for sentiment analysis**
Download the GoEmotions dataset from Kaggle:  
[https://www.kaggle.com/datasets/debarshichanda/goemotions](https://www.kaggle.com/datasets/debarshichanda/goemotions)  
Extract it to `./data/goemotions/`.  
Open and run `bertFineTunning.ipynb` end-to-end to fine-tune the model on this dataset and download the fine tunned model to local desktop for inference.

### **1️⃣ Clone the Repository**

git clone : https://github.com/PrepStation201/mood-based-music-recommendation-system.git

2️⃣ Create and Activate Virtual Environment
# Create
python -m venv venv and 

run bertFinetunning.ipynb file in your local desktop to fine tune the model

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


4️⃣ Download Spotify dataset from Kaggle
[Spotify Dataset – Kaggle](https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset)


5️⃣ Download NLTK Data
python -m nltk.downloader punkt stopwords

6️⃣ Run the Streamlit App
streamlit run app.py


## ⚙️ Project Workflow

1. **User Input** – User enters text describing their mood (e.g., `"I feel happy and excited today!"`).  
2. **Emotion Detection** – Fine-tuned **DistilBERT** analyzes the text and returns emotions (e.g., `['joy', 'excitement']`).  
3. **Candidate Generation (Collaborative)** – **SVD** model generates a broad list of songs for the user ID.  
4. **Mood Matching (Content-Based)** – Creates a target mood vector and ranks candidates by cosine similarity to audio features.  
5. **Contextual Filtering** – Filters recommendations based on the user’s context (e.g., time of day).  
6. **Final Output** – Analyzes sentiments and displays top-ranked, context-aware song recommendations.
