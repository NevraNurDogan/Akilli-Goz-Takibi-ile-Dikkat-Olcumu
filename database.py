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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            duration REAL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS head_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp TEXT,
                direction TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
    cursor.execute("""
           CREATE TABLE IF NOT EXISTS eye_directions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER,
               timestamp TEXT,
               direction TEXT,
               FOREIGN KEY(user_id) REFERENCES users(id)
           )
       """)

    cursor.execute("""
           CREATE TABLE IF NOT EXISTS eye_open_times (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER,
               timestamp TEXT,
               open_duration REAL,
               FOREIGN KEY(user_id) REFERENCES users(id)
           )
       """)

    cursor.execute("""
               CREATE TABLE IF NOT EXISTS distraction_events (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               user_id INTEGER,
               timestamp TEXT,
               reason TEXT,
               FOREIGN KEY(user_id) REFERENCES users(id)

            )
        """)


    conn.commit()
    conn.close()

def save_blink(user_id, blink_count):
    if user_id is None:
        return
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO blinks (user_id, timestamp, blink_count) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), blink_count))


def save_distance(user_id, distance_cm):
    if user_id is None:
        return
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO distances (user_id, timestamp, distance_cm) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), distance_cm))

def save_activity(user_id, duration):
    if user_id is None:
        return
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO sessions (user_id, timestamp, duration) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), duration))

def save_head_position(user_id, pitch, yaw, roll):
    if user_id is None:
        return

    # Baş yönünü metinle ifade et
    if yaw > 15:
        direction = "Sola bakiyor"
    elif yaw < -15:
        direction = "Saga bakiyor"
    elif pitch > 10:
        direction = "Asagi bakiyor"
    elif pitch < -10:
        direction = "Yukari bakiyor"
    else:
        direction = "Duz bakiyor"

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO head_positions (user_id, timestamp, direction) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), direction))


def save_eye_direction(user_id, screen_percentage):
    if user_id is None:
        return
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO eye_directions (user_id, timestamp, direction) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), screen_percentage))

def save_eye_open_time(user_id, open_duration):
    if user_id is None:
        return
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        now = datetime.now(timezone.utc) + timedelta(hours=3)
        cursor.execute("INSERT INTO eye_open_times (user_id, timestamp, open_duration) VALUES (?, ?, ?)",
                       (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), open_duration))


def save_distraction_event(user_id, timestamp=None, reason=""):
    if user_id is None:
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # normalize timestamp -> "YYYY-MM-DD HH:MM:SS" (UTC+3)
    if timestamp is None:
        dt = datetime.now(timezone.utc) + timedelta(hours=3)
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc) + timedelta(hours=3)
    elif isinstance(timestamp, datetime):
        dt = timestamp.astimezone(timezone.utc) + timedelta(hours=3)
    else:
        try:
            # eğer string epoch gelmişse float'a çevir
            f = float(timestamp)
            dt = datetime.fromtimestamp(f, tz=timezone.utc) + timedelta(hours=3)
        except Exception:
            # string olarak sakla (istisnai durum)
            c.execute(
                "INSERT INTO distraction_events (user_id, timestamp, reason) VALUES (?, ?, ?)",
                (user_id, str(timestamp), reason)
            )
            conn.commit()
            conn.close()
            return

    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        "INSERT INTO distraction_events (user_id, timestamp, reason) VALUES (?, ?, ?)",
        (user_id, ts_str, reason)
    )

    conn.commit()
    conn.close()
def get_user_data(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM blinks WHERE user_id = ?", (user_id,))
    blink_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM distances WHERE user_id = ?", (user_id,))
    distance_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,))
    session_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM head_positions WHERE user_id = ?", (user_id,))
    head_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM eye_directions WHERE user_id = ?", (user_id,))
    direction_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM eye_open_times WHERE user_id = ?", (user_id,))
    open_rows = cursor.fetchall()

    cursor.execute("SELECT * FROM distraction_events WHERE user_id = ?", (user_id,))
    distraction_rows = cursor.fetchall()

    conn.close()

    return {
        "blinks": blink_rows,
        "distances": distance_rows,
        "sessions": session_rows,
        "head_positions": head_rows,
        "eye_directions": direction_rows,
        "eye_open_times": open_rows,
        "distractions": distraction_rows

    }

def get_user_role(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

def get_all_users():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users")
        return cursor.fetchall()

def get_user_sessions(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,))
        return cursor.fetchall()

