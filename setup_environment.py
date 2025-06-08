import subprocess
import os
import sys

def install_requirements():
    requirements = [
        'torch',
        'transformers',
        'sentence-transformers',
        'faiss-cpu',
        'bert-score',
        'pyarabic',
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_directories():
    # Create cache directories
    directories = ['./cache', './task1/cache', './task2/cache', './task3/cache']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("\nCreated cache directories")

def main():
    print("Setting up environment for MentalQA project...")
    install_requirements()
    setup_directories()
    print("\nSetup complete!")

if __name__ == "__main__":
    main()
