import requests
import json


def ocr_space_api(image_path, api_key='API-Key', language='kor'):
    url_api = "https://api.ocr.space/parse/image"


    with open(image_path, 'rb') as f:
        response = requests.post(
            url_api,
            files={"filename": f},
            data={"apikey": api_key, "language": language},
            timeout=30
        )


    # JSON으로 파싱
    result = response.json()
 
    parsed = result.get("ParsedResults")
    text_detected = parsed[0].get("ParsedText", "")
    return text_detected


text_result = ocr_space_api('01.jpg')
print("API로부터 추출된 텍스트:")
print(text_result)
