import smtplib
import tkinter as tk
from tkinter import messagebox
from database import setup_database, get_user_data, get_all_users
from detection import Detector
from threading import Thread
import sqlite3
import random
import string
from email.mime.text import MIMEText

def gmail_login():
    return {
        "sender_email": "nevranurdogann@gmail.com",
        "app_password": "sseaofpghdobyuzx"
    }

def send_email(to, subject, message_text):
    creds = gmail_login()
    sender_email = creds["sender_email"]
    app_password = creds["app_password"]

    message = MIMEText(message_text, "plain")
    message["From"] = sender_email
    message["To"] = to
    message["Subject"] = subject

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, to, message.as_string())
        print("Email sent!")
    except Exception as e:
        print(f"Email gönderilemedi: {e}")
        raise e

def generate_password(length=8):#rasgele şifre üretimi
    chars = string.ascii_letters + string.digits + "!@#"
    return ''.join(random.choice(chars) for _ in range(length))

class App:
    def __init__(self, root):
        self.root = root
        self.current_user_id = None
        self.detector = None
        setup_database()
        self.build_login_screen()

    def build_login_screen(self):
        self.root.title("Dikkat Takibi Giriş")
        self.root.geometry("400x300")
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Kullanıcı Adı").pack()
        self.login_name = tk.Entry(self.root)
        self.login_name.pack()

        tk.Label(self.root, text="Şifre").pack()
        self.login_password = tk.Entry(self.root, show="*")
        self.login_password.pack()

        tk.Button(self.root, text="Giriş Yap", command=self.login_user).pack(pady=5)
        tk.Button(self.root, text="Yeni Kullanıcı", command=self.build_register_screen).pack()
        tk.Button(self.root, text="Şifremi Unuttum", command=self.reset_password).pack(pady=5)

    def build_register_screen(self):#yeni kullanıcı kaydı
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Yeni Kullanıcı Adı").pack()
        entry_name = tk.Entry(self.root)
        entry_name.pack()

        tk.Label(self.root, text="Rol Seçin").pack()
        role_var = tk.StringVar(value="user")
        tk.OptionMenu(self.root, role_var, "user", "admin").pack()

        tk.Label(self.root, text="E-posta").pack()
        entry_email = tk.Entry(self.root)
        entry_email.pack()

        def submit_registration():#Formu işleyip kullanıcıyı veritabanına kaydeder.
            name = entry_name.get()
            email = entry_email.get()
            if not name or not email:
                messagebox.showwarning("Eksik", "Tüm alanları doldurun.")
                return
            password = generate_password()
            try:
                send_email(email, "Kayıt Başarılı", f"Merhaba {name},\nŞifreniz: {password}")
            except Exception as e:
                messagebox.showerror("Mail Hatası", f"E-posta gönderilemedi:\n{str(e)}")
                return
            try:
                conn = sqlite3.connect("blink_data4.db")
                cursor = conn.cursor()
                role = role_var.get()
                cursor.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                               (name, email, password, role))

                conn.commit()
                conn.close()
                send_email(email, "Dikkat Takibi Şifreniz", f"Giriş şifreniz: {password}")
                messagebox.showinfo("Başarılı", "Kullanıcı eklendi. Şifre e-postanıza gönderildi.")
                self.build_login_screen()
            except sqlite3.IntegrityError:
                messagebox.showerror("Hata", "Bu kullanıcı adı zaten var!")

        tk.Button(self.root, text="Kaydol", command=submit_registration).pack(pady=5)
        tk.Button(self.root, text="Geri", command=self.build_login_screen).pack()

    def login_user(self):# Kullanıcı adı ve şifreyi kontrol ederek kullanıcıyı giriş yapar.
        name = self.login_name.get()
        password = self.login_password.get()
        conn = sqlite3.connect("blink_data4.db")
        cursor = conn.cursor()
        cursor.execute("SELECT id, role FROM users WHERE name = ? AND password = ?", (name, password))
        result = cursor.fetchone()
        conn.close()
        if result and len(result) == 2:
            self.current_user_id = result[0]
            self.current_user_role = result[1]
            messagebox.showinfo("Giriş", f"Hoş geldin {name}! Rolün: {self.current_user_role}")
            self.build_main()
        else:
            messagebox.showerror("Hatalı Giriş", "Kullanıcı adı veya şifre hatalı.")

    def reset_password(self):#Şifre sıfırlama ekranı oluşturur.
        def send_new_password():
            name = entry_name.get()
            conn = sqlite3.connect("blink_data4.db")
            cursor = conn.cursor()
            cursor.execute("SELECT email FROM users WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result:
                new_password = generate_password()
                cursor.execute("UPDATE users SET password = ? WHERE name = ?", (new_password, name))
                conn.commit()
                send_email(result[0], "Yeni Şifreniz", f"Yeni şifreniz: {new_password}")
                messagebox.showinfo("Şifre Yenilendi", "Yeni şifre e-posta adresinize gönderildi.")
                top.destroy()
            else:
                messagebox.showerror("Hata", "Kullanıcı bulunamadı.")
            conn.close()

        top = tk.Toplevel(self.root)
        top.title("Şifremi Unuttum")
        tk.Label(top, text="Kullanıcı Adınızı Girin").pack()
        entry_name = tk.Entry(top)
        entry_name.pack()
        tk.Button(top, text="Şifreyi Yenile", command=send_new_password).pack(pady=5)

    def build_main(self):#Kullanıcı giriş sonrası ana paneli oluşturur.
        for widget in self.root.winfo_children():
            widget.destroy()
        if self.current_user_role == 'admin':
            tk.Label(self.root, text="Admin Paneli:").pack(pady=5)
            tk.Button(self.root, text="Kullanıcıları Yönet", command=self.admin_user_selection).pack(pady=5)
        else:
            tk.Label(self.root, text="Kullanıcı Paneli").pack(pady=5)

        tk.Button(self.root, text="Başlat", command=self.start_detection).pack(pady=5)
        tk.Button(self.root, text="Durdur", command=self.stop_detection).pack(pady=5)
        tk.Button(self.root, text="Verileri Göster", command=self.show_data).pack(pady=5)
        tk.Button(self.root, text="Şifre Değiştir", command=self.change_password).pack(pady=5)
        back_button = tk.Button(self.root, text="🔙 Geri", command=self.build_login_screen)
        back_button.pack(pady=10)

    def change_password(self):#Şifre değiştirme ekranı:
        def submit_change():
            old_pw = entry_old.get()
            new_pw = entry_new.get()
            if len(new_pw) < 6 or not any(c.isupper() for c in new_pw) or not any(c.islower() for c in new_pw) or not any(c in "!@#" for c in new_pw):
                messagebox.showerror("Hata", "Yeni şifre 6 karakterden uzun, büyük-küçük harf ve özel karakter içermeli.")
                return
            conn = sqlite3.connect("blink_data4.db")
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE id = ?", (self.current_user_id,))
            current_pw = cursor.fetchone()[0]
            if old_pw != current_pw:
                messagebox.showerror("Hata", "Eski şifre yanlış.")
                return
            cursor.execute("UPDATE users SET password = ? WHERE id = ?", (new_pw, self.current_user_id))
            conn.commit()
            conn.close()
            messagebox.showinfo("Başarılı", "Şifre değiştirildi.")
            top.destroy()

        top = tk.Toplevel(self.root)
        top.title("Şifre Değiştir")
        tk.Label(top, text="Eski Şifre").pack()
        entry_old = tk.Entry(top, show="*")
        entry_old.pack()
        tk.Label(top, text="Yeni Şifre").pack()
        entry_new = tk.Entry(top, show="*")
        entry_new.pack()
        tk.Button(top, text="Değiştir", command=submit_change).pack(pady=5)

    def start_detection(self):#Dikkat takibini başlatır.
        self.detector = Detector(self.current_user_id)
        self.detector.start_periodic_save()
        Thread(target=self.detector.start, daemon=True).start()

    def stop_detection(self):#Takibi durdurur.
        if self.detector:
            self.detector.stop()
            messagebox.showinfo("Durdu", "Takip durduruldu.")

    def show_data(self):#Kullanıcının kayıtlı verilerini gösterir.
        blink_rows, distance_rows = get_user_data(self.current_user_id)
        data_window = tk.Toplevel(self.root)
        data_window.title("Veri Kayıt")
        text_area = tk.Text(data_window, width=60, height=30)
        text_area.pack()
        text_area.insert(tk.END, "Göz Kırpma:\n")
        for row in blink_rows:
            text_area.insert(tk.END, f"{row[2]} - Blinks: {row[3]}\n")
        text_area.insert(tk.END, "\nMesafe:\n")
        for row in distance_rows:
            text_area.insert(tk.END, f"{row[2]} - Mesafe: {row[3]:.1f} cm\n")

    def admin_user_selection(self):#Admin kullanıcı için kullanıcı seçme ekranı:
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Kullanıcı Seçimi").pack(pady=5)

        users = get_all_users()
        self.selected_user_id = tk.IntVar(value=self.current_user_id)

        for uid, uname in users:
            rb = tk.Radiobutton(self.root, text=uname, variable=self.selected_user_id, value=uid)
            rb.pack(anchor='w')

        def select_user():
            self.current_user_id = self.selected_user_id.get()
            messagebox.showinfo("Seçildi", f"{[u[1] for u in users if u[0] == self.current_user_id][0]} seçildi.")
            self.build_main()

        tk.Button(self.root, text="Seç", command=select_user).pack(pady=5)
        tk.Button(self.root, text="Geri", command=self.build_main).pack(pady=5)

