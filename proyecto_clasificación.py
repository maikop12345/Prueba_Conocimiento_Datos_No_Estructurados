
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import pytesseract
import cv2
import numpy as np
from nltk.corpus import words
import nltk
from io import BytesIO

# Descargar el corpus de palabras en inglés
nltk.download('words')

app = FastAPI()

def contains_min_english_words_of_length(text, min_count=4, min_length=6):
    english_words = set(words.words())
    text_words = text.lower().split()

    # Contar cuántas palabras en el texto están en el diccionario inglés y tienen al menos min_length caracteres
    long_english_word_count = sum(1 for word in text_words if word in english_words and len(word) >= min_length)

    return long_english_word_count >= min_count

def has_content(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Aplicar OCR para extraer el texto
    extracted_text = pytesseract.image_to_string(gray)

    # Verificar si el texto extraído contiene palabras en inglés de al menos 6 caracteres
    return contains_min_english_words_of_length(extracted_text)

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    
    image = Image.open(BytesIO(await file.read()))
    
    content_present = has_content(image)
    
    return {"filename": file.filename, "content_present": content_present}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)