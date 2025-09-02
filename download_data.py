#!/usr/bin/env python3
"""
Download script for Fake or Real: The Impostor Hunt in Texts competition

This script downloads the competition data using the Kaggle API.
Make sure you have your Kaggle API credentials set up.
"""

import os
import subprocess
import sys

def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_key = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_key):
        return True
    else:
        return False

def download_competition_data():
    """Download the competition data"""
    print("Downloading Fake or Real: The Impostor Hunt in Texts competition data...")
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Change to data directory
        os.chdir('data')
        
        # Download competition data
        cmd = ['kaggle', 'competitions', 'download', '-c', 'fake-or-real-the-impostor-hunt']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Competition data downloaded successfully!")
            
            # Unzip the downloaded file
            import zipfile
            zip_file = 'fake-or-real-the-impostor-hunt.zip'
            if os.path.exists(zip_file):
                print("Extracting files...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')
                
                # Remove zip file
                os.remove(zip_file)
                print("✓ Files extracted successfully!")
                
                # List downloaded files
                print("\nDownloaded files:")
                for root, dirs, files in os.walk('.'):
                    level = root.replace('.', '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"{subindent}{file}")
            else:
                print("⚠ Warning: Downloaded zip file not found")
        else:
            print(f"❌ Error downloading data: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("FAKE OR REAL: THE IMPOSTOR HUNT IN TEXTS")
    print("Data Download Script")
    print("=" * 60)
    
    # Check if Kaggle is installed
    if not check_kaggle_installed():
        print("❌ Kaggle CLI is not installed.")
        print("Please install it using: pip install kaggle")
        print("Or visit: https://github.com/Kaggle/kaggle-api")
        return False
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("❌ Kaggle credentials not configured.")
        print("Please set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✓ Kaggle CLI installed and configured")
    
    # Download data
    if download_competition_data():
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the baseline solution: python src/baseline_solution.py")
        print("2. Explore the data structure")
        print("3. Start experimenting with your models!")
        return True
    else:
        print("\n❌ Download failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
