from PIL import Image  # keep Pillow

def extract_text_from_image(image: Image.Image) -> str:
    """Return OCR text if pytesseract is available; otherwise empty string."""
    try:
        import pytesseract  # lazy import so missing package doesn't crash app
    except Exception:
        return ""  # OCR disabled / not installed

    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""
