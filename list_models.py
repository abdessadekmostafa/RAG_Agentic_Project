import google.generativeai as genai
import os

# Configure la clé API
api_key = os.getenv("GOOGLE_API_KEY", "ma clée api ici")
genai.configure(api_key=api_key)

def list_available_models():
    """
    Liste tous les modèles disponibles pour ton compte API Google Gemini
    et affiche uniquement leur nom.
    """
    try:
        models = genai.list_models()  # Récupère tous les modèles accessibles
        print("Modèles disponibles :\n")
        for m in models:
            print(f"- {m.name}")
    except Exception as e:
        print(f"❌ Une erreur est survenue lors de la récupération des modèles : {e}")

if __name__ == "__main__":
    list_available_models()
