import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pytesseract
import easyocr
from doctr.models import ocr_predictor
from paddleocr import PaddleOCR
from openpyxl import Workbook


import json
import fitz  # PyMuPDF
import torch
import difflib
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pdf2image import convert_from_path
from PIL import Image
from spellchecker import SpellChecker

torch.backends.cudnn.benchmark = True

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'D:\\WorkSpace\\HGS\\Projects\\AppealLetterGeneration\\Models\\TesseractOCR\\tesseract.exe'
performance_file = 'D:\\WorkSpace\\HGS\\Projects\\AppealLetterGeneration\\Data\\OCR_Output\\performance.csv'

def pdf_to_images(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    image_files = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_file = os.path.join(output_folder, f'{os.path.basename(pdf_path)}_page_{page_num + 1}.png')
        pix.save(image_file)
        image_files.append(image_file)
    return image_files

def extract_text_tesseract(image_path):
    return pytesseract.image_to_string(image_path)

def extract_text_doctr(image_path):
    model = ocr_predictor(pretrained=True, assume_straight_pages=True, detect_orientation=True, straighten_pages=True, pretrained_backbone='torch')
    image = Image.open(image_path)
    image_np = np.array(image)
    result = model([image_np])
    text = ""
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text += word.value + " "
    return text.strip()

def extract_text_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    return ' '.join([text for _, text, _ in result])

def extract_text_paddleocr(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.ocr(image_path, cls=True)
    return ' '.join([line[1][0] for line in result[0]])

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compare_texts(text1, text2):
    distance = levenshtein_distance(text1, text2)
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return distance, similarity

def count_spelling_errors(text):

    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    return len(misspelled)

def process_image_file(image_file, models,output_folder):
    results = []
    for model_name, extract_text in models.items():
        start_time = time.time()
        extracted_text = extract_text(image_file)
        end_time = time.time()
        execution_time = end_time - start_time
        word_count = len(extracted_text.split())
        spelling_errors = count_spelling_errors(extracted_text)
        results.append((model_name, image_file, execution_time, word_count, extracted_text, spelling_errors))
        # Save the extracted text to a separate file
        output_text_file = os.path.join(output_folder,
                                        f"{os.path.splitext(image_file)[0]}_{model_name}.txt")
        with open(output_text_file, 'w') as f:
            f.write(extracted_text)

    return results

def process_folder(input_folder, output_folder):
    models = {
        'Tesseract': extract_text_tesseract,
        'Doctr': extract_text_doctr,
        'EasyOCR': extract_text_easyocr,
        'PaddleOCR': extract_text_paddleocr
    }

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['Model', 'File Name', 'Execution Time (s)', 'Word Count','Levenshtein Distance', 'Similarity', 'Spelling Errors'])

    with ProcessPoolExecutor() as executor:
        futures = []
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(input_folder, file_name)
                image_files = pdf_to_images(pdf_path, output_folder)
                for image_file in image_files:
                    futures.append(executor.submit(process_image_file, image_file, models,output_folder))

    # for image_file in os.listdir(input_folder):
    #     pdf_path = os.path.join(input_folder, image_file)
    #     image_files = pdf_to_images(pdf_path, output_folder)
        extracted_texts = {}
        # page_num = 0
        # for image in image_files:
        #     if page_num <= len(image_files)+1:
        #         page_num+=1

        for future in as_completed(futures):
            results = future.result()
            for model_name, image_file, execution_time, word_count, extracted_text, spelling_errors in results:
                file_name = os.path.basename(image_file)
                extracted_texts.setdefault(file_name, {})[model_name] = extracted_text
                sheet.append([model_name, file_name, execution_time, word_count, '', '', spelling_errors])

       # Compare extracted texts between models
        for file_name, texts in extracted_texts.items():
            model_names = list(texts.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1 = model_names[i]
                    model2 = model_names[j]
                    text1 = texts[model1]
                    text2 = texts[model2]
                    distance, similarity = compare_texts(text1, text2)
                    sheet.append([f"{model1} vs {model2}", file_name, '', '', distance, similarity, ''])

    output_file = os.path.join(output_folder, 'ocr_comparison.xlsx')
    workbook.save(output_file)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    input_folder = 'D:\\WorkSpace\\HGS\\Projects\\AppealLetterGeneration\\Data\\MedReports\\'
    output_folder = 'D:\\WorkSpace\\HGS\\Projects\\AppealLetterGeneration\\Data\\OCR_Output\\TestOutput6'
    process_folder(input_folder, output_folder)
