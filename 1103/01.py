from PIL import Image
import pytesseract

img = Image.open('01.jpg')
text = pytesseract.image_to_string(img, lang='kor')

print("추출된 텍스트: ", text)

