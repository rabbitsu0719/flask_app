import Levenshtein


def calculate_accuracy(original_text, ocr_text):
   distance = Levenshtein.distance(original_text, ocr_text)
   max_len = max(len(original_text), len(ocr_text))
   accuracy = (1 - distance / max_len) * 100
   return accuracy


# 예: 원문과 OCR 결과 텍스트
original   = "This is a sample image from the original image."
ocr_result = "This is a smaple image from the original image."


accuracy = calculate_accuracy(original, ocr_result)
print(f"OCR 인식률: {accuracy:.2f}%")