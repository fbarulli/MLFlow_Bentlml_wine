#!/usr/bin/env python3
"""
Check scikit-learn version compatibility with MLflow models
"""
import sklearn
import mlflow
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from mlflow.tracking import MlflowClient

load_dotenv()

print("=" * 60)
print("SKLEARN VERSION COMPATIBILITY CHECK")
print("=" * 60)

# Check current versions
print(f"Current scikit-learn version: {sklearn.__version__}")
print(f"Current MLflow version: {mlflow.__version__}")

# Set up MLflow
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").replace("mlflow+", "")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow URI: {tracking_uri}")

# Try to get model metadata
client = MlflowClient()
models_to_check = [
    "tracking-wine-logisticregression",
    "tracking-wine-randomforest"
]

print(f"\nChecking model compatibility...")

for model_name in models_to_check:
    print(f"\n{model_name}:")
    print("-" * 40)
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            print(f"✅ Latest version: {latest_version.version}")
            
            # Try to load the model
            model_uri = f"models:/{model_name}/{latest_version.version}"
            try:
                model = mlflow.sklearn.load_model(model_uri)
                print(f"✅ Model loads successfully")
                
                # Check if we can make a dummy prediction
                if hasattr(model, 'predict'):
                    # Create dummy data
                    if hasattr(model, 'feature_names_in_'):
                        n_features = len(model.feature_names_in_)
                        feature_names = model.feature_names_in_
                    elif hasattr(model, 'n_features_in_'):
                        n_features = model.n_features_in_
                        feature_names = [f"feature_{i}" for i in range(n_features)]
                    else:
                        n_features = 11  # Common wine dataset features
                        feature_names = [f"feature_{i}" for i in range(n_features)]
                    
                    dummy_data = pd.DataFrame(
                        np.random.random((1, n_features)),
                        columns=feature_names
                    )
                    
                    prediction = model.predict(dummy_data)
                    print(f"✅ Prediction test successful: {prediction}")
                    
            except Exception as load_error:
                print(f"❌ Model loading failed: {str(load_error)}")
                if "version" in str(load_error).lower() or "incompatible" in str(load_error).lower():
                    print(f"⚠️  Version compatibility issue detected!")
                    print(f"   Model was likely trained with scikit-learn 1.2.2")
                    print(f"   Current version is {sklearn.__version__}")
                
        else:
            print(f"❌ No versions found")
            
    except Exception as e:
        print(f"❌ Error checking model: {e}")

print(f"\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print(f"\nIf you're seeing version compatibility issues:")
print(f"1. Downgrade scikit-learn to match training version:")
print(f"   pip install scikit-learn==1.2.2")
print(f"")
print(f"2. Or create a new environment with compatible versions:")
print(f"   conda create -n wine_service python=3.9")
print(f"   conda activate wine_service")
print(f"   pip install -r requirements_compatible.txt")
print(f"")
print(f"3. Or retrain your models with the current scikit-learn version")
print(f"")
print(f"4. Use the version-safe service that handles these errors gracefully")