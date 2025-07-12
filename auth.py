import sqlite3
from utils import generate_password, send_email, is_valid_password

class AuthManager:
    def __init__(self, db_path="users.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT,
                password TEXT,
                role TEXT DEFAULT 'user'
            )
        ''')
        self.conn.commit()

    def register_user(self, username, email):#Aynı kullanıcı adı var mı kontrol eder.
        self.cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if self.cursor.fetchone():
            return False, "Kullanıcı zaten var"

        password = generate_password()
        send_email(email, "Giriş Şifreniz", f"Kullanıcı adınız: {username}\nŞifreniz: {password}")
        self.cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                            (username, email, password))
        self.conn.commit()
        return True, "Kayıt başarılı, şifre mailinize gönderildi."

    def login_user(self, username, password):#Veritabanında bu kullanıcı adı ve şifreyle eşleşen biri var mı kontrol eder.
        self.cursor.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password))
        result = self.cursor.fetchone()

        if result:
         self.current_user_id = result[0]
         self.current_user_role = result[1]

    def change_password(self, username, old_pass, new_pass):#yeni şifre güvenli mi kontrol eder.
        if not is_valid_password(new_pass):
            return False, "Yeni şifre güvenli değil."
        self.cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, old_pass))
        if self.cursor.fetchone():
            self.cursor.execute("UPDATE users SET password=? WHERE username=?", (new_pass, username))
            self.conn.commit()
            return True, "Şifre değiştirildi."
        return False, "Eski şifre yanlış."

    def reset_password(self, email):#Bu e-posta ile kayıtlı kullanıcı var mı kontrol eder.
        self.cursor.execute("SELECT username FROM users WHERE email=?", (email,))
        result = self.cursor.fetchone()
        if not result:
            return False, "E-posta bulunamadı."
        new_pass = generate_password()
        self.cursor.execute("UPDATE users SET password=? WHERE email=?", (new_pass, email))
        self.conn.commit()
        send_email(email, "Yeni Şifreniz", f"Yeni şifreniz: {new_pass}")
        return True, "Yeni şifre mailinize gönderildi."

    def get_all_users(self):#Tüm kullanıcıları listeler.
        self.cursor.execute("SELECT username, email, role FROM users")
        return self.cursor.fetchall()

    def delete_user(self, username):#Kullanıcıyı veritabanından siler.
        self.cursor.execute("DELETE FROM users WHERE username=?", (username,))
        self.conn.commit()
