from typing import Union
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from docx import Document
import os
import tempfile
import logging

# Configuring logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pdf(filepath: str) -> str:
    try:
        reader = PdfReader(filepath)
        text = ''
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text
    except FileNotFoundError:
        logging.error(f"FileNotFoundError: {filepath} not found.")
        raise
    except PdfReadError:
        logging.error(f"PdfReadError: Cannot read {filepath}.")
        raise
    except Exception as e:
        logging.error(f"An unknown error occurred: {e}")
        raise

def load_word_document(filepath: str) -> str:
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist.")
        
        doc = Document(filepath)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(full_text)
    except FileNotFoundError as e:
        logging.error(f"FileNotFoundError: {e}")
        raise
    except Exception as e:
        logging.error(f"An unknown error occurred: {e}")
        raise

def load_file(file: Union[str, bytes]) -> str:
    try:
        new_filepath = ""
        with tempfile.NamedTemporaryFile(delete=False, dir = "temp/") as temp_file:
            temp_file.write(file.getvalue())
            temp_file.seek(0)
            
            file_extension = os.path.splitext(file.name)[-1].lower()
            new_filepath = temp_file.name

            logging.info(f"File path: {file.name}")
            logging.info(f"File extension: {file_extension}")
            
        if file_extension == '.pdf':
            return load_pdf(temp_file.name)
        elif file_extension == '.docx':
            return load_word_document(temp_file.name)
        else:
            logging.warning("Unsupported file type")
            return "Unsupported file type"
    except Exception as e:
        logging.error(f"Error processing the file: {e}")
        return f"Error processing the file: {e}"
