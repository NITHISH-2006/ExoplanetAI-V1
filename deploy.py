#!/usr/bin/env python3
"""
One-click deployment script for Exoplanet Detection AI
Run: python deploy.py
"""

import os
import subprocess
import sys
import shutil

def setup_project():
    """Setup complete project structure"""
    print("🚀 Setting up Exoplanet Detection AI...")
    
    # Create directories
    directories = ['src', 'webapp', 'models']
    for dir in directories:
        os.makedirs(dir, exist_ok=True)
        print(f"✅ Created {dir}/")
    
    # Create .streamlit config
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write("""
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
""")
    
    print("✅ Project structure created!")
    
    # Install requirements
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed!")
    except Exception as e:
        print(f"❌ Dependency installation failed: {e}")
        print("Please run: pip install -r requirements.txt")
    
    # Train initial model
    print("🤖 Training initial model...")
    try:
        from src.data_generator import RealisticDataGenerator
        from src.model import LightweightDetector
        
        data_gen = RealisticDataGenerator()
        model = LightweightDetector()
        
        dataset = data_gen.generate_dataset(n_samples=300)
        model.train(dataset['flux'], dataset['labels'], dataset['periods'])
        model.save('models/model.joblib')
        
        print("✅ Model trained and saved!")
    except Exception as e:
        print(f"⚠️ Model training skipped: {e}")
    
    print("\n🎉 SETUP COMPLETE!")
    print("\n🚀 NEXT STEPS:")
    print("1. Test locally: streamlit run webapp/app.py")
    print("2. Deploy to Streamlit Cloud:")
    print("   - Push to GitHub")
    print("   - Go to share.streamlit.io")
    print("   - Connect repository and deploy!")
    print("\n📱 Your app will be live in minutes!")

if __name__ == "__main__":
    setup_project()