import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import random

class MusicEmotionClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.emotion_categories = ['happy', 'sad', 'energetic', 'calm', 'aggressive']
        
    def preprocess_data(self, spotify_data_path):
        """Preprocess Spotify metadata for training"""
        # Load Spotify metadata
        df = pd.read_excel(spotify_data_path)
        
        # Extract audio features
        features = []
        for _, row in df.iterrows():
            # For demo purposes, generate random audio features
            # In production, these would come from Spotify API
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
            print(f"Processed track: {row['track_name']}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        
        # Initially, we'll use a rule-based approach to assign emotion labels
        # This can be replaced with actual labeled data when available
        emotions = self._assign_initial_emotions(features_df)
        
        return features_df, emotions
    
    def _assign_initial_emotions(self, features):
        """Assign initial emotion labels based on audio features"""
        emotions = []
        
        for _, row in features.iterrows():
            # Simple rule-based classification
            if row['energy'] > 0.7 and row['valence'] > 0.7:
                emotion = 'happy'
            elif row['energy'] < 0.4 and row['valence'] < 0.4:
                emotion = 'sad'
            elif row['energy'] > 0.8 and row['loudness'] > -5:
                emotion = 'energetic'
            elif row['energy'] < 0.4 and row['acousticness'] > 0.6:
                emotion = 'calm'
            elif row['energy'] > 0.7 and row['loudness'] > -4 and row['valence'] < 0.4:
                emotion = 'aggressive'
            else:
                # Default to the most common category
                emotion = 'energetic'
                
            emotions.append(emotion)
        
        # One-hot encode the emotions
        emotions_df = pd.get_dummies(emotions)
        
        # Ensure all emotion categories are present
        for emotion in self.emotion_categories:
            if emotion not in emotions_df.columns:
                emotions_df[emotion] = 0
                
        # Make sure to return a numpy array with the columns in the correct order
        return emotions_df[self.emotion_categories].astype(float).values
    
    def build_model(self, input_shape):
        """Build the neural network model for emotion classification"""
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.emotion_categories), activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the model on the preprocessed data"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy:.4f}")
        
        return history
    
    def save_model(self, model_path='emotion_classifier_model.h5', scaler_path='emotion_scaler.pkl'):
        """Save the trained model and scaler"""
        if self.model is not None:
            self.model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path} and scaler saved to {scaler_path}")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, model_path='emotion_classifier_model.h5', scaler_path='emotion_scaler.pkl'):
        """Load a trained model and scaler"""
        from keras.models import load_model
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path} and scaler loaded from {scaler_path}")
    
    def predict_emotion(self, features):
        """Predict emotion from audio features"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        
        # Get emotion labels
        predicted_emotions = [self.emotion_categories[np.argmax(pred)] for pred in predictions]
        
        return predicted_emotions, predictions

# Example usage
if __name__ == "__main__":
    classifier = MusicEmotionClassifier()
    
    # Process Spotify data
    print("Processing Spotify metadata...")
    X, y = classifier.preprocess_data('spotify_metadata.xlsx')
    
    # Train model
    print("Training model...")
    classifier.train(X, y, epochs=30)
    
    # Save model
    classifier.save_model()
    
    print("Emotion classifier training complete!")
