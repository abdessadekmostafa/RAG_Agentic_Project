from fpdf import FPDF
import datetime
import unicodedata

# Fonction pour nettoyer le texte de tout caractère non-ASCII
def clean_text(text):
    if not text:
        return ""
    # Normaliser les caractères Unicode en forme de compatibilité
    text = unicodedata.normalize('NFKD', text)
    # Remplacer les caractères spéciaux par leur équivalent ASCII
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Supprimer tous les caractères qui ne sont pas ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def generate_pdf_history(history, pdf_names):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "RAG Agentic - Résumé de session", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)

    pdf.cell(0, 10, f"Documents analysés : {', '.join(pdf_names)}", ln=True)
    pdf.ln(5)

    for i, item in enumerate(history, 1):
        question_text = clean_text(item.get('question', ''))
        answer_text = clean_text(item.get('answer', ''))

        pdf.multi_cell(0, 10, f"Q{i}: {question_text}", ln=True)
        pdf.multi_cell(0, 10, f"R{i}: {answer_text}", ln=True)
        pdf.ln(5)

    pdf_file = f"RAG_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(pdf_file)
    return pdf_file
