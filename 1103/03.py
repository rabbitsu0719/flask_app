# PDF에서 텍스트 추출

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io


def extract_text_from_pdf(pdf_path):
   doc = fitz.open(pdf_path)
   full_text = ""


   for page_num in range(len(doc)):
       page = doc.load_page(page_num)
       text = page.get_text()
       full_text += text


       # 이미지 추출
       image_list = page.get_images()
       for img_index, img in enumerate(image_list):
           xref = img[0]
           base_image = doc.extract_image(xref)
           image_bytes = base_image["image"]
           img_pil = Image.open(io.BytesIO(image_bytes))


           # OCR 수행하여 이미지 내 텍스트 추출
           img_text = pytesseract.image_to_string(img_pil, lang='kor')
           full_text += "\n[이미지 내 텍스트]\n" + img_text + "\n"


   return full_text


pdf_text = extract_text_from_pdf("sample.pdf")
print("PDF 텍스트 및 이미지 내 텍스트:")
print(pdf_text)



