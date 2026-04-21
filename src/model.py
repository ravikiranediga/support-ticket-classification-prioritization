"""
Model Module
ML Pipeline: TF-IDF + Linear SVM for ticket classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.svm import LinearSVC

from config import (
    MODELS_DIR, REPORTS_DIR,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TEST_SIZE, RANDOM_STATE
)


class TicketClassifier:
    """
    Classifier for ticket type or priority.
    Uses TF-IDF + Linear SVM pipeline.
    """

    def __init__(self, target_name: str):
        """
        Parameters
        ----------
        target_name : str
            Column name to predict (e.g., 'Ticket Type' or 'Ticket Priority')
        """
        self.target_name = target_name
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.is_trained = False

    def _build_pipeline(self):
        """Create TF-IDF + SVM pipeline"""
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=2,
            max_df=0.95,
            stop_words='english',
            sublinear_tf=True
        )
        clf = LinearSVC(random_state=RANDOM_STATE, dual=False, C=1.0, max_iter=2000)

        return Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])

    def train(self, df: pd.DataFrame, text_column: str = 'combined_text'):
        """
        Train the model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with target column and text
        text_column : str
            Name of column containing preprocessed text

        Returns
        -------
        dict : Training results (accuracy, f1, etc.)
        """
        print(f"\n{'='*60}")
        print(f"TRAINING: {self.target_name}")
        print(f"{'='*60}")

        # Prepare X and y
        X = df[text_column].values
        y = df[self.target_name].values

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_

        print(f"Total samples: {len(X)}")
        print(f"Classes: {self.classes_}")
        for cls, cnt in zip(*np.unique(y, return_counts=True)):
            print(f"  {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )
        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # Train
        print("\nTraining Linear SVM...")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes_))

        # Confusion matrix
        self._save_confusion_matrix(y_test, y_pred)

        # Save model
        self._save_model()

        self.is_trained = True
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _save_confusion_matrix(self, y_true, y_pred):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.label_encoder.classes_
        )
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')
        plt.title(f'{self.target_name} - Confusion Matrix')
        plt.tight_layout()

        safe_name = self.target_name.replace(' ', '_').lower()
        save_path = REPORTS_DIR / f'confusion_matrix_{safe_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved: reports/confusion_matrix_{safe_name}.png")

    def _save_model(self):
        """Save pipeline and encoder to disk"""
        safe_name = self.target_name.replace(' ', '_').lower()
        joblib.dump(self.pipeline, MODELS_DIR / f'{safe_name}_pipeline.pkl')
        joblib.dump(self.label_encoder, MODELS_DIR / f'{safe_name}_encoder.pkl')
        print(f"Model saved: models/{safe_name}_pipeline.pkl")

    def predict(self, subject: str, description: str, preprocess_func):
        """
        Predict for a single ticket.

        Parameters
        ----------
        subject : str
            Ticket subject line
        description : str
            Ticket description text
        preprocess_func : callable
            Function to combine and preprocess text

        Returns
        -------
        dict : {'prediction': str, 'confidence': float or None}
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        combined = preprocess_func(subject, description)
        pred_encoded = self.pipeline.predict([combined])[0]
        prediction = self.label_encoder.inverse_transform([pred_encoded])[0]

        # Get confidence score (SVM decision function)
        confidence = None
        if hasattr(self.pipeline.named_steps['clf'], 'decision_function'):
            scores = self.pipeline.decision_function([combined])[0]
            confidence = float(max(scores))

        return {'prediction': prediction, 'confidence': confidence}

    @classmethod
    def load(cls, target_name: str):
        """
        Load trained model from disk.

        Parameters
        ----------
        target_name : str
            Either 'Ticket Type' or 'Ticket Priority'

        Returns
        -------
        TicketClassifier : Loaded model instance
        """
        safe_name = target_name.replace(' ', '_').lower()
        model = cls(target_name)
        model.pipeline = joblib.load(MODELS_DIR / f'{safe_name}_pipeline.pkl')
        model.label_encoder = joblib.load(MODELS_DIR / f'{safe_name}_encoder.pkl')
        model.classes_ = model.label_encoder.classes_
        model.is_trained = True
        return model
