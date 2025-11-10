from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('upload.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)

from PIL import Image
import pytesseract
import re
from flask import jsonify

# 1. 텍스트 추출 함수
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 3 -c preserve_interword_spaces=1')
    return os.linesep.join([s for s in text.splitlines() if s])

# 2. 영수증 정보 파싱
def parse_receipt(text):
    lines = text.splitlines()
    store_name = lines[0].strip()  # 첫 번째 줄을 상호명으로 가정

    business_id_match = re.search(r'(사업자(등록)?번호\s*[:\s]*(\d{3}[-.\s]?\d{2}[-.\s]?\d{5}))', text)
    business_id = business_id_match.group(3).replace(' ', '').replace('-', '').replace('.', '') if business_id_match else "정보 없음"

    amount_match = re.search(r'(총액|합계|결제금액)\s*[:\s]*(\d{1,3}(?:,\d{3})*원?)', text)
    total_amount = amount_match.group(2) if amount_match else "정보 없음"

    return {"상호명": store_name, "사업자번호": business_id, "금액": total_amount}

# 3. JSON 응답
@app.route('/upload_and_ocr', methods=['POST'])
def upload_and_ocr():
    # 파일 처리 및 텍스트 추출
    text = extract_text_from_image(filepath)  # filepath는 업로드된 이미지의 경로로 설정되어야 함

    # 영수증 정보 파싱
    parsed_data = parse_receipt(text)

    # JSON 형태로 응답
    return jsonify(parsed_data)
# 업로드 폴더 및 허용된 확장자 설정
app.config['UPLOAD_FOLDER'] = 'uploads/'

# 최대 파일 크기 (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 허용되는 이미지 확장자 목록 (화이트리스트)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 파일 확장자 검증 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 파일 저장 함수
def save_file(file):
    # 안전한 파일명 생성
    filename = secure_filename(file.filename)
    
    # 파일 저장 경로 생성
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 파일 저장
    file.save(filepath)




