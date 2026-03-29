# setup.py
import os
import sys
import subprocess

def setup():
    print("="*60)
    print("MEDICINE RECOMMENDATION SYSTEM - SETUP")
    print("="*60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Check if data files exist
    data_files = [
        'data/description.csv',
        'data/diets.csv',
        'data/medications.csv',
        'data/precautions_df.csv',
        'data/Symptom-severity.csv',
        'data/symtoms_df.csv'
    ]
    
    print("\nChecking data files...")
    missing_files = []
    for file in data_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ ERROR: Missing data files!")
        print("Please ensure all CSV files are in the 'data/' folder.")
        return
    
    # Install dependencies
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed")
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    print("✓ Created models directory")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run 'python train.py' to train the model")
    print("2. Run 'python run.py' to start the application")
    print("="*60)

if __name__ == "__main__":
    setup()