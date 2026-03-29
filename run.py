# run.py
import sys
import os
import tkinter as tk

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("STARTING MEDICINE RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Check if model exists
    model_path = 'models/disease_model.h5'
    if not os.path.exists(model_path):
        print("\n⚠️  Model not found!")
        print("Please run 'python train.py' first to train the model.")
        print("\nWould you like to train the model now? (y/n): ", end="")
        response = input().lower()
        
        if response == 'y':
            import train
            train.main()
        else:
            print("\nExiting...")
            return
    
    # Run GUI
    print("\n🚀 Launching GUI...")
    root = tk.Tk()
    
    # Import here to avoid circular imports
    from gui.main_window import MedicineRecommendationGUI
    app = MedicineRecommendationGUI(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()