#!/usr/bin/env python3
"""
Script to check available models and their stages in MLflow Model Registry
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").replace("mlflow+", "")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

# Initialize MLflow client
client = MlflowClient()

# Model names to check
model_names = [
    "tracking-wine-logisticregression",
    "tracking-wine-randomforest"
]

print("=" * 60)
print("CHECKING MODEL REGISTRY")
print("=" * 60)

for model_name in model_names:
    print(f"\nModel: {model_name}")
    print("-" * 40)
    
    try:
        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            print(f"❌ No versions found for model '{model_name}'")
            continue
            
        print(f"✅ Found {len(model_versions)} version(s)")
        
        for version in model_versions:
            print(f"  Version {version.version}:")
            print(f"    - Current Stage: {version.current_stage}")
            print(f"    - Aliases: {version.aliases}")
            print(f"    - Status: {version.status}")
            print(f"    - Run ID: {version.run_id}")
            
            # Check if this version is in Staging
            if "Staging" in version.aliases or version.current_stage == "Staging":
                print(f"    ✅ Available in Staging")
            else:
                print(f"    ❌ NOT in Staging")
                
    except Exception as e:
        print(f"❌ Error checking model '{model_name}': {e}")

print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

print("\nIf models exist but are not in 'Staging' stage, you can:")
print("1. Use the latest version instead of 'Staging'")
print("2. Set a version to 'Staging' stage using MLflow UI")
print("3. Use aliases to assign 'Staging' to a specific version")

print("\nTo set a model version to Staging programmatically:")
print("```python")
print("# Set version to Staging stage")
print("client.transition_model_version_stage(")
print("    name='tracking-wine-logisticregression',")
print("    version='1',  # Replace with actual version")
print("    stage='Staging'")
print(")")
print("")
print("# OR set Staging alias")
print("client.set_registered_model_alias(")
print("    name='tracking-wine-logisticregression',")
print("    alias='Staging',")
print("    version='1'  # Replace with actual version")
print(")")
print("```")