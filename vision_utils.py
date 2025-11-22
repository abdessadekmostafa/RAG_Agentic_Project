import fitz
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Charger le modèle BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_images_in_pdf(pdf_path):
    """
    Génère des descriptions textuelles des images contenues dans un PDF
    en utilisant BLIP (image captioning).
    """
    doc = fitz.open(pdf_path)
    descriptions = []

    for page_num, page in enumerate(doc):
        # Extraire les images de la page
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # RGB
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Générer une description avec BLIP
            inputs = processor(pil_img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            descriptions.append(f"[IMAGE_DESCRIPTION] Page {page_num+1}, Image {img_index+1}: {caption}")

    return descriptions
