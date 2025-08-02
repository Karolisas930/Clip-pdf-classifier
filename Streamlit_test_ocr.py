import streamlit as st
from ocr_utils import extract_text_from_image

text = extract_text_from_image("some_image.png")
st.write("Extracted text:", text)
