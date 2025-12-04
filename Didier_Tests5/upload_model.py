import kagglehub
import os
from dotenv import load_dotenv

env_path = "./config/.env"
load_dotenv(dotenv_path=env_path)

# Configuration
# Path to the local folder containing adapter_model.bin/safetensors and adapter_config.json
# LOCAL_MODEL_PATH = "preference_class_gemma_2"
LOCAL_MODEL_PATH = "preference_class_gemma_1b_fix_2"

# Kaggle Handle: <your-username>/<model-slug>/<framework>/<variation-slug>
MODEL_HANDLE = "didiersalazar/gemma-3-preference-adapter/transformers/v7"

print(f"Logging in to Kaggle...")

# Need to have the kaggle api key in ~/.kaggle/kaggle.json
# Download from Kaggle, from the legacy API key and put it in the directory
# kagglehub.login()

print(f"Uploading model from '{LOCAL_MODEL_PATH}' to '{MODEL_HANDLE}'...")

try:
    # This will create the model if it doesn't exist, or create a new version if it does
    kagglehub.model_upload(
        handle=MODEL_HANDLE,
        local_model_dir=LOCAL_MODEL_PATH,
        license_name="Apache 2.0",  # Standard license
        version_notes="Seventh Fine-tuned LoRA adapter for LMSYS competition using Gemma 3 1B IT (1 Epoch, trained with swapped augmented data, 97.5/2.5 train/val, Changed prompt to handle rounds of prompts/responses)"
    )
    print("\n-> Upload successful!")
except Exception as e:
    print(f"\n-> Upload failed: {e}")