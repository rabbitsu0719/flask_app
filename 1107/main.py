from flask import Flask, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

users = {
    'user1': 'password1',
    'user2': 'password2'
}

@app.route('/')
def home():
    if 'username' in session:
        print(f"로그인 상태: {session['username']}")
        return f"안녕하세요, {session['username']}님! <a href='/logout'>로그아웃</a>"
    else:
        print("로그인 안됨")
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            print(f"{username} 로그인 성공")
            return redirect(url_for('home'))
        else:
            print(f"로그인 실패: 아이디 또는 비밀번호가 틀림")
            return '''
                아이디 또는 비밀번호가 틀렸습니다.<br><a href="/login">다시 로그인</a>
            '''
    return '''
        <form method="post">
            사용자명: <input type="text" name="username"><br>
            비밀번호: <input type="password" name="password"><br>
            <input type="submit" value="로그인">
        </form>
    '''

@app.route('/logout')
def logout():
    username = session.pop('username', None)
    print(f"{username} 로그아웃")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

