from langchain.tools import Tool
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import pymupdf
import os
import re
from typing import Optional, Tuple
import numpy as np

ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def extract_title_text_from_pdf(pdf_path: str, region: Optional[Tuple[int, int, int, int]] = None):
    """
    Extracts text from the first page of a PDF bridge plan using OCR.
    Provide the full file path of the PDF.
    """
    try:
        # --- Method 1: Fast Text Extraction with PyMuPDF ---
        doc = pymupdf.open(pdf_path)
        first_page = doc.load_page(0)  # Load the first page
        if region:
            clip = pymupdf.Rect(*region) # unpack coordinates
            text = first_page.get_text("text", clip=clip)
        else:
            text = first_page.get_text("text")
        doc.close()

        # If we got a reasonable amount of text, return it.
        if text and len(text.strip()) > 20:
            print("Successfully extracted text directly from PDF.")
            return text
        
        # --- Method 2: Fallback to OCR for Scanned Images ---
        print("Direct text extraction failed or insufficient. Falling back to OCR.")
        # Convert first page to image
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi = 300)
        img = images[0]
        if region:
            # Crop image to specified region
            img = img.crop(region)
        # temp_image_path = "temp_title_page.png"
        # img.save(temp_image_path)
        img_array = np.array(img)

        # Run OCR
        print("Running OCR on title page image...")
        ocr_result = ocr_model.predict(img_array)
        # os.remove(temp_image_path)

        if ocr_result:
            for res in ocr_result:
                if "rec_texts" in res:
                    full_text = "\n".join(res["rec_texts"])
            return full_text
        else:
            return "OCR processing returned no results."
        
    except Exception as e:
        return f"OCR extraction failed: {str(e)}"
    
ocr_tool = Tool(
    name = "extract_text_from_title_sheet",
    func = lambda pdf_path: extract_title_text_from_pdf(pdf_path, region=(2550, 1650, 5100, 3300)),
    description = "extract title sheet text from pdf",
)

def filter_target_section(text: str) -> str:
    """
    Light preprocessing only. Normalize OCR text and add hints for GPT parsing.
    """
    # Fix OCR squished words (like CONTRACTFOR)
    text = re.sub(r"(CONTRACT)\s*FOR", r"CONTRACT FOR", text, flags=re.IGNORECASE)

    # Insert a colon and a space between CONTRACT FOR and next uppercase word block if squished
    text = re.sub(r"(CONTRACT FOR)([A-Z])", r"\1: \2", text, flags=re.IGNORECASE)

    # Normalize Date and Job Number markers
    text = re.sub(r"\bJN\b", " Job Number ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDATE\b", " Date ", text, flags=re.IGNORECASE)

    # Remove obvious noise
    text = re.sub(r"TITLE SHEET|DRAWING|SHEET|TITLE|DESIGN", "", text, flags=re.IGNORECASE)

    # Remove Structure Number including formats: B01-22222, S09-3, S09-3 of 22222 and C03 of 22222
    text = re.sub(r"\b[A-Z]\d{2}(?:-\d{1,5}(?:\s+of\s+\d{5})?|\s+of\s+\d{5})\b", "", text, flags=re.IGNORECASE)

    # Collapse excessive whitespace
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

filter_tool = Tool (
    name = "filter_target_section",
    func = filter_target_section,
    description = "Filter and return only the target from raw OCR text."
)