from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

cd Clip-pdf-classifier
git add ocr_utils.py test_ocr.py
git commit -m "Add OCR utility and test script"
git push origin main # or `master` depending on branch
