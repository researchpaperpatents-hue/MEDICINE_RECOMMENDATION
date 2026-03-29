# src/model_trainer.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
import os

class ModelTrainer:
    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.history = None
        os.makedirs(model_path, exist_ok=True)
    
    def build_model(self, input_shape, num_classes):
        """Build neural network model"""
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the model"""
        from sklearn.model_selection import train_test_split
        
        print(f"Training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Build model
        self.model = self.build_model(X.shape[1], len(np.unique(y)))
        
        # Print model summary
        self.model.summary()
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation accuracy: {test_acc:.4f}")
        
        return self.history
    
    def save_model(self, filename='disease_model.h5'):
        """Save trained model"""
        full_path = os.path.join(self.model_path, filename)
        self.model.save(full_path)
        print(f"Model saved to {full_path}")
    
    def load_model(self, filename='disease_model.h5'):
        """Load trained model"""
        full_path = os.path.join(self.model_path, filename)
        self.model = keras.models.load_model(full_path)
        print(f"Model loaded from {full_path}")
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1), np.max(predictions, axis=1)
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_history.png'))
        plt.show()