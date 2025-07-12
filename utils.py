import smtplib
import string
from random import choice
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def gmail_authenticate():
    return {
        "sender_email": "nevranurdogann@gmail.com",
        "app_password": "sseaofpghdobyuzx"
    }

def generate_password(length=8):
    chars = string.ascii_letters + string.digits + "!@#"
    return ''.join(choice(chars) for _ in range(length))


def send_email(receiver_email, subject, body):
    creds = gmail_authenticate()
    sender_email = creds["sender_email"]
    app_password = creds["app_password"]

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    html_content = f"""\
    <html>
      <body>
        <p>{body}</p>
      </body>
    </html>
    """

    message.attach(MIMEText(html_content, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("✅ Mail başarıyla gönderildi.")
        return True
    except Exception as e:
        print(f'❌ Mail gönderilemedi: {e}')
        return False
import re

def is_valid_password(password):
    if len(password) < 6:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#]', password):
        return False
    return True
