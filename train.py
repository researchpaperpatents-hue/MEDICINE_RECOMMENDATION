# train.py
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
import matplotlib.pyplot as plt

def main():
    print("="*60)
    print("MEDICINE RECOMMENDATION SYSTEM - TRAINING")
    print("="*60)
    
    try:
        # Process data
        print("\n1. Processing data...")
        processor = DataProcessor()
        X, y = processor.load_and_process()
        processor.save_encoders()
        
        print(f"\n   ✓ Data processed successfully")
        print(f"     - Number of samples: {len(X)}")
        print(f"     - Number of features: {X.shape[1]}")
        print(f"     - Number of diseases: {len(processor.unique_diseases)}")
        
        # Train model
        print("\n2. Training neural network...")
        trainer = ModelTrainer()
        history = trainer.train(X, y, epochs=100, batch_size=32)
        
        # Save model
        trainer.save_model()
        
        # Plot results
        print("\n3. Generating training plots...")
        trainer.plot_training_history()
        
        # Final evaluation
        final_acc = history.history['val_accuracy'][-1]
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE!")
        print(f"  Final validation accuracy: {final_acc:.4f}")
        print("="*60)
        print("\nYou can now run the GUI application:")
        print("  python run.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()