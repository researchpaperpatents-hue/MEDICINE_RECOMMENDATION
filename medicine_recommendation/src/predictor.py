# src/predictor.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class DiseasePredictor:
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.data_processor = None
        self.is_loaded = False
    
    def initialize(self, data_processor):
        """Initialize predictor with data processor"""
        self.data_processor = data_processor
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        model_file = os.path.join(self.model_path, 'disease_model.h5')
        
        if os.path.exists(model_file):
            self.model = keras.models.load_model(model_file)
            self.is_loaded = True
            print("Model loaded successfully")
        else:
            print("Model not found. Please train the model first.")
            self.is_loaded = False
    
    def predict(self, symptoms):
        """Predict disease from symptoms"""
        if not self.is_loaded:
            raise Exception("Model not loaded. Please train or load model first.")
        
        if not symptoms:
            raise Exception("No symptoms provided")
        
        # Get feature vector
        feature_vector = self.data_processor.get_symptom_vector(symptoms)
        
        # Predict
        predictions = self.model.predict(feature_vector.reshape(1, -1), verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        # Get disease name
        disease = self.data_processor.idx_to_disease[predicted_idx]
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_predictions = [
            (self.data_processor.idx_to_disease[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        return {
            'disease': disease,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': predictions.tolist()
        }
    
    def get_recommendations(self, disease):
        """Get recommendations for a disease"""
        if not self.data_processor:
            raise Exception("Data processor not initialized")
        
        return self.data_processor.get_disease_info(disease)