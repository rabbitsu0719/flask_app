import easyocr
import numpy as np
from PIL import Image

# pip install easyocr

def read_korean_text(image_path, min_confidence=0.5):
   # EasyOCR 리더 초기화 (처음 실행시 모델 다운로드 됨)
   reader = easyocr.Reader(['ko', 'en'], gpu=True)
  
   # 이미지 읽기
   try:
       result = reader.readtext(image_path)
   except Exception as e:
       return f"Error reading image: {str(e)}"
  
   # 결과 출력 준비
   detected_texts = []
  
   # 각 탐지된 텍스트 처리
   for (bbox, text, prob) in result:
       if prob >= min_confidence:  # 신뢰도가 기준값 이상인 것만 처리
           detected_texts.append({
               'text': text,
               'confidence': prob,
               'bbox': bbox
           })
  
   return detected_texts

if __name__ == "__main__":
   # 이미지 경로
   image_path = "1103/01.jpg"
  
   print("텍스트 인식 시작...")
   results = read_korean_text(image_path)
  
   if isinstance(results, str):  # 에러 메시지인 경우
       print(results)