from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import os
from pathlib import Path

def test_hf_token():
    # Load token from .env
    env_path = Path('.env')
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path.absolute()}")
        return False
        
    load_dotenv()
    
    # Get token from environment
    token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not token:
        print("Error: No HuggingFace token found in .env")
        return False
    else:
        print(f"Found token starting with: {token[:8]}...")
        
    try:
        # Initialize API and test token
        api = HfApi(token=token)
        user = api.whoami()
        print(f"Success! Logged in as: {user}")
        
        # Test model access
        print("\nTesting model access...")
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        try:
            model_info = api.model_info(model_id)
            print(f"Model access OK: {model_info.modelId}")
            print(f"Model tags: {model_info.tags}")
            print(f"Model access: {model_info.gated}")
        except HfHubHTTPError as e:
            print(f"\nError accessing model: {e}")
            print("\nPlease visit:")
            print(f"https://huggingface.co/{model_id}")
            print("And accept the model terms of use.")
        
        return True
    except Exception as e:
        print(f"Error testing token: {e}")
        return False

if __name__ == "__main__":
    test_hf_token() 