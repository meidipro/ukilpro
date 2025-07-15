import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

# Configure the generative AI library with the API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# --- Function to list available models ---
def list_available_models():
    print("Listing available Gemini models...")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"  {m.name} (supports generateContent)")
    except Exception as e:
        print(f"Error listing models: {e}")
    print("----------------------------------")

# Call the function to list models
list_available_models()
