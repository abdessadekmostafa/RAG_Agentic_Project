import pdfplumber
import fitz
from PIL import Image
import pytesseract
import tempfile
from vision_utils import describe_images_in_pdf

# Chemin vers Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(uploaded_file):
    """
    Extraction hybride :
    - Texte brut avec pdfplumber
    - Tableaux convertis en texte structuré
    - OCR sur pages (PyMuPDF)
    - Description des images (BLIP)
    """
    text = ""

    # Sauvegarde du fichier uploadé dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # --- Texte + tableaux ---
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join([", ".join(row) for row in table if row])
                text += "\n[TABLE]\n" + table_text + "\n"

    # --- OCR sur pages entières ---
    doc = fitz.open(tmp_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text.strip():
            text += "\n[IMAGE_OCR]\n" + ocr_text + "\n"

    # --- Description des images avec BLIP ---
    descriptions = describe_images_in_pdf(tmp_path)
    for desc in descriptions:
        text += "\n" + desc + "\n"

    return text


def split_text_into_chunks(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
