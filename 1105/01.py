# Tesseract OCR을 사용하여 이미지에서 한국어 텍스트 추출

from PIL import Image
import pytesseract
import cv2

#img = Image.open('1103/01.jpg') # 150dpi 이상의 해상도 권장

# 전처리: 이미지 개선 작업 가능 (예: 그레이스케일 변환, 이진화 등)
def preprocess_receipt_image(image_path, resized_width=800):

   # 이미지 흑백(grayscale)으로 읽기
   image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   # 가로 크기를 resized_width로 맞추고 종횡비 유지하며 리사이즈
   height, width = image.shape
   ratio = resized_width / width
   dim = (resized_width, int(height * ratio))
   resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

   # 명암 대비 향상을 위한 히스토그램 평활화 적용
   equalized = cv2.equalizeHist(resized)

   # 결과 이미지 저장 또는 반환
   return equalized

preprocessed_img = preprocess_receipt_image('1103/01.jpg')
img = Image.fromarray(preprocessed_img)

# OCR 수행
text = pytesseract.image_to_string(img, lang='kor')
# 후처리: 정규표현식 등을 사용하여 불필요한 문자 제거 가능

print("추출된 텍스트: ", text)