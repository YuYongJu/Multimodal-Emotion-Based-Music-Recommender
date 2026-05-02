"""Music emotion classifier.

Reads real audio features (Spotify schema) from the metadata DataFrame
and trains a small Keras MLP. The previous version fabricated features
with random.uniform() at both training and inference time, so the model
had no predictive signal — that has been removed.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]

EMOTION_CATEGORIES = ["happy", "sad", "energetic", "calm", "aggressive"]


class MissingFeaturesError(ValueError):
    """Raised when the metadata DataFrame doesn't have audio features.

    Run `python main.py --fetch-spotify` first to produce a metadata file
    with real audio features attached.
    """


class MusicEmotionClassifier:
    def __init__(self) -> None:
        self.model = None
        self.scaler = StandardScaler()
        self.emotion_categories = EMOTION_CATEGORIES

    def _features_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise MissingFeaturesError(
                f"audio feature columns missing from metadata: {missing}. "
                "This metadata file predates audio_features integration; "
                "regenerate it with `python main.py --fetch-spotify`."
            )
        return df[FEATURE_COLUMNS].copy()

    def preprocess_data(self, spotify_data_path: str):
        df = pd.read_excel(spotify_data_path)
        features_df = self._features_from_dataframe(df)
        features_df = features_df.dropna()
        if features_df.empty:
            raise MissingFeaturesError(
                "All rows have at least one NaN audio feature. Aborting."
            )
        emotions = self._assign_initial_emotions(features_df)
        return features_df, emotions

    def _assign_initial_emotions(self, features: pd.DataFrame) -> np.ndarray:
        """Rule-based labels over real features.

        These rules are deliberately simple and explainable, and they
        produce a labelled dataset that the neural net then learns to
        approximate. The neural net's value over the rules is smoother
        decision boundaries on the continuous feature space and the
        ability to retrain on properly-labelled data later.
        """
        emotions = []
        for _, row in features.iterrows():
            energy, valence = row["energy"], row["valence"]
            loudness, acousticness = row["loudness"], row["acousticness"]
            if energy > 0.7 and valence > 0.7:
                emotions.append("happy")
            elif energy < 0.4 and valence < 0.4:
                emotions.append("sad")
            elif energy > 0.7 and loudness > -5 and valence < 0.4:
                emotions.append("aggressive")
            elif energy > 0.7 and loudness > -7:
                emotions.append("energetic")
            elif energy < 0.5 and acousticness > 0.5:
                emotions.append("calm")
            else:
                emotions.append("energetic" if energy >= 0.5 else "calm")

        encoded = pd.get_dummies(emotions)
        for emotion in self.emotion_categories:
            if emotion not in encoded.columns:
                encoded[emotion] = 0
        return encoded[self.emotion_categories].astype(float).values

    def build_model(self, input_shape: int):
        model = Sequential([
            Dense(64, activation="relu", input_dim=input_shape),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
            Dense(len(self.emotion_categories), activation="softmax"),
        ])
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model = model
        return model

    def train(self, X, y, epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=validation_split, random_state=42
        )
        if self.model is None:
            self.build_model(X_train.shape[1])
        history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_test, y_test), verbose=1,
        )
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Validation loss: {loss:.4f} | Validation accuracy: {accuracy:.4f}")
        return history

    def save_model(self, model_path: str = "emotion_classifier_model.h5",
                   scaler_path: str = "emotion_scaler.pkl") -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved model → {model_path}, scaler → {scaler_path}")

    def load_model(self, model_path: str = "emotion_classifier_model.h5",
                   scaler_path: str = "emotion_scaler.pkl") -> None:
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Loaded model from {model_path}, scaler from {scaler_path}")

    def predict_emotion(self, features: pd.DataFrame):
        if self.model is None:
            raise RuntimeError(
                "Model not trained or loaded — call train() or load_model() first"
            )
        if isinstance(features, pd.DataFrame):
            features = features[FEATURE_COLUMNS].values
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled, verbose=0)
        labels = [self.emotion_categories[int(np.argmax(p))] for p in predictions]
        return labels, predictions


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "spotify_metadata.xlsx"
    classifier = MusicEmotionClassifier()
    X, y = classifier.preprocess_data(path)
    classifier.train(X, y, epochs=30)
    classifier.save_model()
