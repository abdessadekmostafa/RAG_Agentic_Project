import streamlit as st
from utils import extract_text_from_pdf, split_text_into_chunks
from rag import create_faiss_index, agentic_rag
from pdf_utils import generate_pdf_history

st.set_page_config(page_title="RAG Agentic PDF", page_icon="🤖", layout="wide")
st.title("📄 RAG Agentic avec Gemini + FAISS")

# Historique
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Upload multiple PDFs
uploaded_pdfs = st.file_uploader(
    "Téléverse un ou plusieurs PDF :", 
    type="pdf", 
    accept_multiple_files=True
)

all_chunks = []

if uploaded_pdfs:
    for pdf in uploaded_pdfs:
        with st.spinner(f"Extraction du texte depuis {pdf.name}..."):
            text = extract_text_from_pdf(pdf)

        st.markdown(f"### Aperçu du texte extrait : {pdf.name}")
        st.text_area(f"Texte extrait de {pdf.name}", text[:1000], height=200)

        chunks = split_text_into_chunks(text)
        all_chunks.extend(chunks)

    if all_chunks:
        index, chunks = create_faiss_index(all_chunks)
        st.success("Indexation réussie ! ✔")

# Question
query = st.text_input("Pose une question aux documents :")

if query and uploaded_pdfs:
    with st.spinner("Analyse agentic + RAG en cours..."):
        answer = agentic_rag(query, index, chunks)

    st.markdown("### 🧠 Réponse :")
    st.write(answer)

    # Ajout à l’historique
    st.session_state.qa_history.append({"question": query, "answer": answer})

# Historique
if st.session_state.qa_history:
    st.markdown("### 📜 Historique :")
    for item in reversed(st.session_state.qa_history):
        st.markdown(f"**Q :** {item['question']}")
        st.markdown(f"**R :** {item['answer']}")
        st.write("---")

# Génération PDF de l’historique
if st.button("Générer PDF de l'historique"):
    pdf_file = generate_pdf_history(st.session_state.qa_history, [pdf.name for pdf in uploaded_pdfs])
    with open(pdf_file, "rb") as f:
        st.download_button("Télécharger le PDF", f, file_name=pdf_file)
