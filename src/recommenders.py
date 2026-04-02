# src/recommenders.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    """
    Handles SVD-based collaborative filtering.
    Trains on user interaction data and generates personalized recommendations.
    """
    def __init__(self, user_ratings_df, all_songs_df):
        self.all_songs_df = all_songs_df
        self.ratings_df = user_ratings_df
        self.svd_model = SVD()
        self._train()
        print("✅ CollaborativeRecommender initialized and SVD model trained.")

    def _train(self):
        """Private method to train the SVD model."""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userID', 'songID', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd_model.fit(trainset)

    def recommend(self, user_id, n=200):
        """Gets top N song recommendations for a user."""
        all_song_ids = self.all_songs_df['song_id'].unique()
        interacted_songs = self.ratings_df[self.ratings_df['userID'] == user_id]['songID'].unique()
        
        predictions = []
        for song_id in all_song_ids:
            if song_id not in interacted_songs:
                predictions.append(self.svd_model.predict(user_id, song_id))
        
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_n_ids = [pred.iid for pred in predictions[:n]]
        
        return self.all_songs_df[self.all_songs_df['song_id'].isin(top_n_ids)].copy()


class HybridRecommender:
    """
    The main engine that combines all recommender components.
    """
    def __init__(self, emotion_detector, collab_recommender, data_manager):
        self.emotion_detector = emotion_detector
        self.collab_recommender = collab_recommender
        self.data_manager = data_manager
        
        self.mood_filters = {
            'admiration': {'valence': 0.8, 'energy': 0.6},
            'amusement': {'valence': 0.8, 'energy': 0.7},
            'anger': {'valence': 0.2, 'energy': 0.8},
            'annoyance': {'valence': 0.3, 'energy': 0.6},
            'approval': {'valence': 0.7, 'energy': 0.5},
            'caring': {'valence': 0.7, 'energy': 0.4},
            'confusion': {'valence': 0.4, 'energy': 0.5},
            'curiosity': {'valence': 0.6, 'energy': 0.6},
            'desire': {'valence': 0.7, 'energy': 0.6},
            'disappointment': {'valence': 0.3, 'energy': 0.3},
            'disapproval': {'valence': 0.3, 'energy': 0.5},
            'disgust': {'valence': 0.2, 'energy': 0.6},
            'embarrassment': {'valence': 0.3, 'energy': 0.4},
            'excitement': {'valence': 0.8, 'energy': 0.8},
            'fear': {'valence': 0.3, 'energy': 0.7},
            'gratitude': {'valence': 0.8, 'energy': 0.5},
            'grief': {'valence': 0.1, 'energy': 0.2},
            'joy': {'valence': 0.8, 'energy': 0.7},
            'love': {'valence': 0.9, 'energy': 0.6},
            'nervousness': {'valence': 0.4, 'energy': 0.7},
            'optimism': {'valence': 0.8, 'energy': 0.6},
            'pride': {'valence': 0.8, 'energy': 0.7},
            'realization': {'valence': 0.6, 'energy': 0.6},
            'relief': {'valence': 0.7, 'energy': 0.4},
            'remorse': {'valence': 0.2, 'energy': 0.3},
            'sadness': {'valence': 0.2, 'energy': 0.3},
            'surprise': {'valence': 0.6, 'energy': 0.8},
            'neutral': {'valence': 0.5, 'energy': 0.5}
        }

    def _apply_contextual_filter(self, recommendations, context):
        """Applies filters based on user context (e.g., time of day)."""
        if not context:
            return recommendations

        if context.get('time_of_day') == 'evening':
            return recommendations[recommendations['energy'] < 0.6]
        if context.get('activity') == 'workout':
            return recommendations[recommendations['energy'] > 0.7]

        return recommendations

    def recommend(self, user_text, user_id, context=None, top_n=10):
        """
        Generates hybrid recommendations.
        """
        # 1. Detect mood from text
        emotions = self.emotion_detector.predict(user_text)

        # 2. Get collaborative recommendations as a starting point
        collab_recs = self.collab_recommender.recommend(user_id, n=200)

        # 3. Re-rank based on mood (Content-Based Filtering)
        audio_features = self.data_manager.audio_features
        # Start with the dataset average for all features, so we don't accidentally ask for 0 tempo, 0 speechiness, etc
        target_features = self.data_manager.df[audio_features].mean().copy()
        
        emotions_found = [em for em in emotions if em in self.mood_filters]
        if not emotions_found:
            emotions_found = ['neutral']
        
        target_val = 0.0
        target_eng = 0.0
        for emotion in emotions_found:
            target_val += self.mood_filters[emotion]['valence']
            target_eng += self.mood_filters[emotion]['energy']
        
        # Override the valence and energy with the mood-specific attributes
        target_features['valence'] = target_val / len(emotions_found)
        target_features['energy'] = target_eng / len(emotions_found)
        
        # Calculate cosine similarity
        recs_features_scaled = self.data_manager.scaler.transform(collab_recs[audio_features])
        target_features_scaled = self.data_manager.scaler.transform(pd.DataFrame([target_features]))
        
        similarity = cosine_similarity(recs_features_scaled, target_features_scaled).flatten()
        collab_recs['mood_match_score'] = similarity
        
        mood_filtered_recs = collab_recs.sort_values(
            by=['mood_match_score', 'popularity'], ascending=[False, False]
        )
        
        # 4. Apply contextual filtering
        final_recs = self._apply_contextual_filter(mood_filtered_recs, context)
        
        return final_recs.head(top_n)