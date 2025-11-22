import os
import json
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------
# 1. EMBEDDINGS
# ---------------------------
def get_embedding(text):
    model = "models/text-embedding-004"
    result = genai.embed_content(model=model, content=text)
    return np.array(result["embedding"], dtype="float32")


# ---------------------------
# 2. CRÉATION INDEX FAISS
# ---------------------------
def create_faiss_index(chunks):
    embeddings = [get_embedding(chunk) for chunk in chunks]
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, chunks


# ---------------------------
# 3. AGENT : PLANIFICATION
# ---------------------------
def agentic_planner(question):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Tu es un agent RAG autonome.
    Analyse la question et renvoie un plan JSON contenant :
    - "query_reformulee": la question optimisée pour la recherche vectorielle
    - "k": le nombre optimal de chunks à récupérer (entre 3 et 8)
    - "strategie": "large" ou "precis"

    Question utilisateur : {question}

    Format JSON strict :
    {{
        "query_reformulee": "",
        "k": 3,
        "strategie": ""
    }}
    """

    result = model.generate_content(prompt).text

    try:
        return json.loads(result)
    except:
        return {"query_reformulee": question, "k": 3, "strategie": "precis"}


# ---------------------------
# 4. RÉCUPÉRATION CONTEXTE
# ---------------------------
def retrieve_context(query, index, chunks, k):
    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]


# ---------------------------
# 5. GÉNÉRATION DE RÉPONSE
# ---------------------------
def generate_answer(question, context):
    context_text = "\n".join(context)

    prompt = f"""
    Tu es un système RAG.
    Utilise EXCLUSIVEMENT le contexte ci-dessous pour répondre.

    CONTEXTE :
    {context_text}

    QUESTION : {question}

    RÉPONSE :
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


# ---------------------------
# 6. SELF-CHECK
# ---------------------------
def self_check(question, answer):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Vérifie si la réponse répond bien à la question.
    Répond par : "OK" ou "RECHERCHER_PLUS".

    Question : {question}
    Réponse : {answer}
    """
    decision = model.generate_content(prompt).text.strip()
    return decision


# ---------------------------
# 7. PIPELINE AGENTIC RAG
# ---------------------------
def agentic_rag(question, index, chunks):
    # Étape 1 : Plan
    plan = agentic_planner(question)

    # Étape 2 : Recherche
    context = retrieve_context(plan["query_reformulee"], index, chunks, plan["k"])

    # Étape 3 : Générer première réponse
    answer = generate_answer(question, context)

    # Étape 4 : Vérification
    decision = self_check(question, answer)

    if "RECHERCHER_PLUS" in decision:
        plan["k"] = min(plan["k"] + 3, 10)
        context = retrieve_context(plan["query_reformulee"], index, chunks, plan["k"])
        answer = generate_answer(question, context)

    return answer
