import fitz  # PyMuPDF
from io import BytesIO
from typing import List
from fastapi import UploadFile

class PDFService:
    async def extract_text(self, pdf_bytes: bytes) -> str:
        """Extrait le texte d'un PDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            return " ".join([page.get_text() for page in doc])
        except Exception as e:
            raise RuntimeError(f"Erreur d'extraction PDF : {str(e)}")

    async def process_uploaded_cvs(self, files: List[UploadFile]) -> dict:
        """Traite tous les CVs uploadés et retourne un dictionnaire ID => texte"""
        results = {}
        for file in files:
            try:
                # Read the uploaded file content
                pdf_bytes = await file.read()
                text = await self.extract_text(pdf_bytes)
                results[file.filename] = {
                    "filename": file.filename,
                    "text": text
                }
            except Exception as e:
                print(f"Erreur avec {file.filename} : {str(e)}")
                results[file.filename] = {
                    "filename": file.filename,
                    "error": str(e)
                }
            finally:
                # Ensure the file pointer is reset (important if you need to read again)
                await file.seek(0)
        return results