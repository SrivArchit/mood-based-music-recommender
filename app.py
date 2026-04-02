import streamlit as st
import pandas as pd
import numpy as np

# Import your custom classes from the src directory
from src.data_manager import DataManager
from src.emotion_detector import EmotionDetector
from src.recommenders import CollaborativeRecommender, HybridRecommender

# --- Page Configuration ---
st.set_page_config(
    page_title="Mood-Based Recommender",
    page_icon="🎶",
    layout="centered"
)

# --- Caching: Load all models and data only once ---
@st.cache_resource
def load_resources():
    """
    Loads and initializes all the necessary classes and data for the app.
    This function is cached to ensure it only runs once.
    """
    # --- Configuration ---
    MODEL_PATH = "./fine-tuned-goemotions-model"
    TOKENIZER_NAME = "distilbert-base-uncased"
    SPOTIFY_DATA_PATH = "./Data/data/dataset.csv"

    # 1. Initialize the DataManager
    data_manager = DataManager(SPOTIFY_DATA_PATH)
    
    # 2. Initialize the EmotionDetector
    emotion_detector = EmotionDetector(MODEL_PATH, TOKENIZER_NAME)

    # 3. Simulate user data and initialize the CollaborativeRecommender
    # In a real app, this ratings_df would come from a database.
    ratings_df = pd.DataFrame({
        'userID': np.random.randint(0, 500, 1_000_000),
        'songID': np.random.choice(data_manager.get_all_data()['song_id'], 1_000_000),
        'rating': np.random.randint(1, 6, 1_000_000)
    })
    collab_recommender = CollaborativeRecommender(ratings_df, data_manager.get_all_data())
    
    # 4. Initialize the main HybridRecommender with all components
    hybrid_recommender = HybridRecommender(
        emotion_detector=emotion_detector,
        collab_recommender=collab_recommender,
        data_manager=data_manager
    )
    
    return hybrid_recommender

# --- Load all assets using the cached function ---
hybrid_recommender = load_resources()

# --- Streamlit User Interface ---
st.title("🎶 Hybrid Mood-Based Recommender")
st.write("Enter how you're feeling, select a user ID, and get personalized song recommendations.")

# --- User Input ---
user_text = st.text_input("How are you feeling right now?", "I am feeling happy and excited today!")
sample_user_id = st.selectbox("Select a User ID (for personalization)", (42, 101, 256, 312, 499))
context_option = st.selectbox("Select a Context (Optional)", ("None", "Evening", "Workout"))

# --- This is the NEW code block ---
if st.button("Get Recommendations"):
    if user_text:
        # --- MINIMAL CHANGE STARTS HERE ---
        # 1. Detect emotions immediately for instant feedback
        detected_emotions = hybrid_recommender.emotion_detector.predict(user_text)
        st.info(f"🧠 **Detected Mood(s):** {', '.join(detected_emotions)}")
        # --- MINIMAL CHANGE ENDS HERE ---

        # Determine context from user selection
        context = None
        if context_option == 'Evening':
            context = {'time_of_day': 'evening'}
        elif context_option == 'Workout':
            context = {'activity': 'workout'}
        
        # Get recommendations (this will re-run detection internally, which is fine for one user)
        with st.spinner('Finding songs that match your mood...'):
            recommendations = hybrid_recommender.recommend(
                user_text=user_text,
                user_id=sample_user_id,
                context=context
            )
        
        st.success("Here are your personalized recommendations!")
        st.dataframe(recommendations[['name', 'artists', 'valence', 'energy', 'mood_match_score']])
    else:
        st.error("Please enter some text to describe your mood.")