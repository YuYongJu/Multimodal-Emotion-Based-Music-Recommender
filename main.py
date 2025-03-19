#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from AutoLabel import MusicEmotionClassifier
import argparse
import sys
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from GoogleVideoIntelligenceAPI import analyze_videos_in_bucket
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Initialize Spotify client with environment variables
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
))

# Set Google Cloud credentials path from environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Get playlist IDs from environment variable
PLAYLIST_IDS = os.getenv('PLAYLIST_IDS').split(',')

# Get bucket name from environment variables
BUCKET_NAME = os.getenv('BUCKET_NAME', 'music-emotion-classification-videos')

def fetch_spotify_data(playlist_id=None):
    """Fetch Spotify metadata for a playlist"""
    print("Fetching Spotify metadata...")
    from recommend_spotify_playlist_music_for_tiktok_edits import fetch_spotify_metadata
    
    if not playlist_id:
        # Use default playlist ID
        playlist_id = '65LdqYCLcsV0lJoxpeQ6fW'
    
    # Fetch metadata
    df = fetch_spotify_metadata(playlist_id)
    
    # Export the dataframe to an Excel file
    output_file = 'spotify_metadata.xlsx'
    df.to_excel(output_file, index=False)
    
    print(f"Spotify data exported to {output_file}")
    return output_file

def analyze_video(bucket_name):
    """Analyze video content using Google Cloud Video Intelligence API"""
    print("Analyzing video content...")
    
    try:
        # Try to import and run the actual video analysis
        video_data_path = analyze_videos_in_bucket(bucket_name)
        return video_data_path
    except Exception as e:
        print(f"Error with Google Cloud Video Intelligence: {e}")
        print("Using mock video analysis data for demonstration...")
        
        # Create mock video analysis data
        return create_mock_video_analysis()
        
def create_mock_video_analysis():
    """Create mock video analysis data for demonstration purposes"""
    # Sample data that might come from video analysis
    data = {
        'Video': ['sample_video.mp4'] * 25,
        'Label Description': [
            'dance', 'music', 'performance', 'concert', 'singing',
            'party', 'fun', 'entertainment', 'crowd', 'stage',
            'night', 'light', 'colorful', 'excitement', 'joy',
            'energy', 'movement', 'happy', 'young', 'group',
            'song', 'artist', 'band', 'festival', 'celebration'
        ],
        'Category Description': [
            'Entertainment', 'Music', 'Performance', 'Art', 'Event',
            'Concert', 'Culture', 'Dance', 'Leisure', 'Nightlife',
            'Social', 'Recreation', 'Festival', 'Celebration', 'Group',
            'Activity', 'Fun', 'Performing Arts', 'Stage', 'Audience',
            'Show', 'Live', 'Party', 'Crowd', 'Lighting'
        ],
        'Start Time': [i * 5.0 for i in range(25)],
        'End Time': [i * 5.0 + 4.0 for i in range(25)],
        'Confidence': [0.95, 0.92, 0.88, 0.85, 0.83,
                      0.80, 0.79, 0.77, 0.76, 0.75,
                      0.74, 0.72, 0.71, 0.70, 0.69,
                      0.68, 0.67, 0.66, 0.65, 0.64,
                      0.63, 0.62, 0.61, 0.60, 0.59]
    }
    
    # Create a DataFrame and save to Excel
    df = pd.DataFrame(data)
    output_file = 'GoogleVideoIntelligenceLabelAnalyzer_results.xlsx'
    
    # Create Excel writer
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='Label Detection', index=False)
    
    print(f"Mock video analysis data saved to {output_file}")
    return output_file

def train_emotion_classifier(spotify_data_path):
    """Train the music emotion classifier"""
    print("Training music emotion classifier...")
    
    # Create classifier
    classifier = MusicEmotionClassifier()
    
    # Process data
    X, y = classifier.preprocess_data(spotify_data_path)
    
    # Train model
    classifier.train(X, y, epochs=30)
    
    # Save model
    classifier.save_model()
    
    print("Emotion classifier training complete!")
    return classifier

def recommend_music_for_video(video_data_path='GoogleVideoIntelligenceLabelAnalyzer_results.xlsx', 
                        spotify_data_path='spotify_metadata.xlsx',
                        model_path='emotion_classifier_model.h5'):
    """Recommend music for a video based on its content"""
    print("Recommending music for video...")
    
    # Default video data if file doesn't exist (for testing)
    if not os.path.exists(video_data_path):
        # Create sample data for testing
        print("Creating sample video data for testing...")
        video_data = {
            'Label Description': ['anime', 'mangaka', 'illustration', 'song', 'flame'] * 5,
            'Category Description': ['artwork', 'person', 'art', 'music', 'fire'] * 5,
            'Confidence': [0.9, 0.8, 0.7, 0.6, 0.5] * 5
        }
        video_df = pd.DataFrame(video_data)
    else:
        # Load video analysis data
        video_df = pd.read_excel(video_data_path)
    
    # Extract dominant labels and categories
    label_counts = video_df["Label Description"].value_counts()
    category_counts = video_df["Category Description"].value_counts()
    
    top_labels = label_counts.head(5).index.tolist()
    top_categories = category_counts.head(5).index.tolist()
    
    print(f"Top video labels: {', '.join(top_labels)}")
    print(f"Top video categories: {', '.join(top_categories)}")
    
    # Determine target emotions based on video content
    target_emotions = map_video_content_to_emotions(top_labels, top_categories)
    print(f"Target emotions based on video content: {', '.join(target_emotions)}")
    
    # Load music data
    music_df = pd.read_excel(spotify_data_path)
    
    # Load classifier
    classifier = MusicEmotionClassifier()
    if os.path.exists(model_path):
        try:
            classifier.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            classifier = train_emotion_classifier(spotify_data_path)
    else:
        print("No pretrained model found. Training new model...")
        classifier = train_emotion_classifier(spotify_data_path)
    
    # Extract features for prediction using random values (for demo)
    features = []
    for _, row in music_df.iterrows():
        feature_dict = {
            'danceability': random.uniform(0, 1),
            'energy': random.uniform(0, 1),
            'key': random.randint(0, 11),
            'loudness': random.uniform(-60, 0),
            'mode': random.randint(0, 1),
            'speechiness': random.uniform(0, 1),
            'acousticness': random.uniform(0, 1),
            'instrumentalness': random.uniform(0, 1),
            'liveness': random.uniform(0, 1),
            'valence': random.uniform(0, 1),
            'tempo': random.uniform(50, 200),
            'duration_ms': row.get('duration_ms', 0),
            'popularity': row.get('popularity', 0)
        }
        features.append(feature_dict)
    
    features_df = pd.DataFrame(features)
    
    # Predict emotions for each track
    predicted_emotions, emotion_scores = classifier.predict_emotion(features_df)
    
    # Create recommendations dataframe
    recommendations = pd.DataFrame({
        'track_name': music_df['track_name'],
        'artist': music_df['artist'],
        'predicted_emotion': predicted_emotions
    })
    
    # Calculate emotion match scores
    match_scores = []
    for i, emotion in enumerate(predicted_emotions):
        score = 0
        if emotion in target_emotions:
            # Base score for matching emotion
            score = 100
            
            # Bonus for popularity
            popularity = music_df.iloc[i].get('popularity', 0)
            score += min(popularity, 30)  # Max 30 points for popularity
            
            # Bonus for energy match with video intensity
            avg_confidence = video_df['Confidence'].mean()
            energy = features_df.iloc[i]['energy']
            energy_match = 1 - abs(avg_confidence - energy)
            score += energy_match * 20  # Max 20 points for energy match
        
        match_scores.append(score)
    
    recommendations['match_score'] = match_scores
    
    # Sort by match score
    recommended_tracks = recommendations.sort_values('match_score', ascending=False).head(10)
    
    print("\nTop recommended tracks for your video:")
    for i, (_, track) in enumerate(recommended_tracks.iterrows(), 1):
        print(f"{i}. {track['track_name']} by {track['artist']} - {track['predicted_emotion']} (Score: {track['match_score']:.1f})")
    
    return recommended_tracks

def map_video_content_to_emotions(labels, categories):
    """Map video content to target emotions"""
    emotion_mapping = {
        # Common labels
        'dance': ['energetic', 'happy'],
        'performance': ['energetic'],
        'music': ['happy', 'energetic'],
        'fun': ['happy'],
        'smile': ['happy'],
        'nature': ['calm'],
        'water': ['calm'],
        'sky': ['calm'],
        'fight': ['aggressive'],
        'explosion': ['aggressive'],
        'romance': ['calm', 'sad'],
        'love': ['happy', 'calm'],
        'food': ['happy'],
        'sports': ['energetic'],
        'game': ['energetic'],
        'cry': ['sad'],
        'night': ['calm', 'sad'],
        'sunset': ['calm'],
        'party': ['happy', 'energetic'],
        
        # Categories
        'Entertainment': ['happy', 'energetic'],
        'Sports': ['energetic'],
        'Art': ['calm'],
        'Nature': ['calm'],
        'Action': ['energetic', 'aggressive'],
        'Drama': ['sad', 'calm'],
        'Comedy': ['happy'],
        'Adventure': ['energetic'],
        'Romance': ['calm', 'sad']
    }
    
    # Default emotions if no matches
    target_emotions = set(['energetic', 'happy'])
    
    # Find matches in labels
    for label in labels:
        label_lower = label.lower()
        for key, emotions in emotion_mapping.items():
            if key.lower() in label_lower:
                for emotion in emotions:
                    target_emotions.add(emotion)
    
    # Find matches in categories
    for category in categories:
        category_lower = category.lower()
        for key, emotions in emotion_mapping.items():
            if key.lower() in category_lower:
                for emotion in emotions:
                    target_emotions.add(emotion)
    
    return list(target_emotions)

def main():
    parser = argparse.ArgumentParser(description='Music Emotion Classification and Video Recommendation System')
    parser.add_argument('--fetch-spotify', action='store_true', help='Fetch Spotify data')
    parser.add_argument('--playlist-id', type=str, help='Spotify playlist ID to fetch')
    parser.add_argument('--analyze-video', action='store_true', help='Analyze video content')
    parser.add_argument('--bucket-name', type=str, default='anime_food_landscape_object_bucket', help='Google Cloud bucket name for videos')
    parser.add_argument('--train-model', action='store_true', help='Train emotion classifier model')
    parser.add_argument('--recommend', action='store_true', help='Recommend music for video')
    parser.add_argument('--full-pipeline', action='store_true', help='Run the full pipeline')
    args = parser.parse_args()
    
    spotify_data_path = 'spotify_metadata.xlsx'
    video_data_path = 'GoogleVideoIntelligenceLabelAnalyzer_results.xlsx'
    
    # Run full pipeline if requested
    if args.full_pipeline:
        args.fetch_spotify = True
        args.analyze_video = True
        args.train_model = True
        args.recommend = True
    
    # Fetch Spotify data
    if args.fetch_spotify:
        spotify_data_path = fetch_spotify_data(args.playlist_id)
    
    # Analyze video
    if args.analyze_video:
        video_data_path = analyze_video(args.bucket_name)
    
    # Train emotion classifier
    if args.train_model:
        train_emotion_classifier(spotify_data_path)
    
    # Recommend music for video
    if args.recommend:
        recommend_music_for_video(video_data_path, spotify_data_path)
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 