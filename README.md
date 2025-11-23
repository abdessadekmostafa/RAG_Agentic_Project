# RAG PDF Project

Un projet de **Retrieval-Augmented Generation (RAG)** permettant d’ingérer des documents PDF et d’en extraire :
- ✅ Texte brut
- ✅ Tableaux
- ✅ Texte dans les images (OCR via Tesseract)
- ✅ Description des images sans texte (BLIP Vision)
- ✅ Découpage en chunks et indexation dans FAISS
- ✅ Interface utilisateur avec Streamlit pour poser des questions

---

## Fonctionnalités
- Extraction hybride (texte + tableaux + images).
- OCR avec **Tesseract** pour lire le texte dans les images scannées.
- Génération de descriptions d’images avec **BLIP**.
- Découpage en chunks pour une recherche efficace.
- Indexation vectorielle avec **FAISS**.
- Interface interactive avec **Streamlit**.
