# This class will manage loading and preparing the Spotify dataset.
# Load, clean, and preprocess the song data.

# src/data_manager.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataManager:
    """
    Handles loading, cleaning, and preprocessing of the Spotify song dataset.
    """
    def __init__(self, spotify_data_path):
        """
        Initializes the DataManager by loading and preparing the data.

        Args:
            spotify_data_path (str): The file path to the Spotify data CSV.
        """
        # Load and clean the main dataset
        self.df = pd.read_csv(spotify_data_path)
        # Handle different column naming conventions across Spotify datasets
        if 'track_name' in self.df.columns and 'name' not in self.df.columns:
            self.df = self.df.rename(columns={'track_name': 'name'})
        # Drop unnamed index column if present
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(columns=['Unnamed: 0'])
        self.df = self.df.dropna(subset=['name', 'artists']).copy()

        # ----------------------------------------------------
        # FILTER: Keep only English and Hindi (Indian) songs
        # ----------------------------------------------------
        if 'track_genre' in self.df.columns:
            # Common western genres + 'indian' for Hindi tracks
            target_genres = [
                'indian', 'pop', 'rock', 'hip-hop', 'dance', 'acoustic', 
                'alt-rock', 'alternative', 'blues', 'country', 'disco', 
                'electronic', 'folk', 'heavy-metal', 'indie', 'indie-pop', 
                'jazz', 'metal', 'punk', 'r-n-b', 'soul', 'synth-pop'
            ]
            self.df = self.df[self.df['track_genre'].isin(target_genres)].copy()
            
            # Optional: Reset index just in case the filtering messes up the ID order heavily
            self.df = self.df.reset_index(drop=True)
        self.df['song_id'] = self.df.index # Create a unique ID for each song

        # Define the audio features to be used for content-based filtering
        self.audio_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness', 
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]

        # Initialize and fit the scaler for audio features
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.df[self.audio_features])
        
        print("✅ DataManager initialized: Spotify data loaded and scaled.")

    def get_song_by_id(self, song_id):
        """Returns a song's data by its unique song_id."""
        return self.df[self.df['song_id'] == song_id]

    def get_all_data(self):
        """Returns the complete, cleaned DataFrame."""
        return self.df