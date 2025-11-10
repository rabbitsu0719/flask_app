import matplotlib.pyplot as plt

image_names = ['image1', 'image2', 'image3']
accuracies = [92.5, 85.3, 89.1]

plt.bar(image_names, accuracies)
plt.title("OCR 인식률 비교")
plt.xlabel("이미지이름")
plt.ylabel("인식률(%)")
plt.ylim(0, 100)

plt.savefig("ocr_result.png")
print("✅ 그래프 저장 완료: ocr_result.png")


