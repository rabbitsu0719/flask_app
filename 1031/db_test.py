import mysql.connector

# MySQL 데이터베이스 연결
conn = mysql.connector.connect(
    host="localhost",        # MySQL 서버 주소
    user="kakao_user",       # 사용자명
    password="1234",         # 비밀번호
    database="kakao"         # 사용할 데이터베이스 이름
)

# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SHOW TABLES;")  # 현재 DB 내 테이블 목록 보기

# 결과 출력
result = cursor.fetchall()
print("✅ 현재 kakao 데이터베이스의 테이블 목록:")
for row in result:
    print(row)

# 커서와 연결 종료
cursor.close()
conn.close()
