import sqlite3
from datetime import datetime, timezone, timedelta

DB_NAME = "blink_data4.db"

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blinks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            blink_count INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS distances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            distance_cm REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

def save_blink(user_id, blink_count):
    if user_id is None:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    formatted_time = datetime.now(timezone.utc) + timedelta(hours=3)
    cursor.execute("INSERT INTO blinks (user_id, timestamp, blink_count) VALUES (?, ?, ?)",
                   (user_id, formatted_time.strftime("%Y-%m-%d %H:%M:%S"), blink_count))
    conn.commit()
    conn.close()

def save_distance(user_id, distance_cm):
    if user_id is None:
        return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    formatted_time = datetime.now(timezone.utc) + timedelta(hours=3)
    cursor.execute("INSERT INTO distances (user_id, timestamp, distance_cm) VALUES (?, ?, ?)",
                   (user_id, formatted_time.strftime("%Y-%m-%d %H:%M:%S"), distance_cm))
    conn.commit()
    conn.close()

def get_user_data(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM blinks WHERE user_id = ?", (user_id,))
    blink_rows = cursor.fetchall()
    cursor.execute("SELECT * FROM distances WHERE user_id = ?", (user_id,))
    distance_rows = cursor.fetchall()
    conn.close()
    return blink_rows, distance_rows

def get_user_role(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_all_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users")
    users = cursor.fetchall()
    conn.close()
    return users

