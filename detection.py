import logging
import os
import sqlite3
import threading
import cv2
import dlib
import time
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from threading import Thread
from plyer import notification
from datetime import datetime

from database import (
    save_blink,
    save_distance,
    save_head_position,
    save_eye_direction,
    save_eye_open_time,
    save_activity,
    save_distraction_event

)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Göz ve iris landmark indeksleri (MediaPipe Face Mesh Refined Iris modelinden)
LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173,
                133, 155, 154, 153, 145, 144, 163, 7]  # soldaki göz çevresi
RIGHT_EYE_IDX = [362, 398, 384, 385, 386, 387, 388, 466,
                 263, 249, 390, 373, 374, 380, 381, 382]  # sağdaki göz çevresi

LEFT_IRIS_IDX = 468  # sol iris merkeazi
RIGHT_IRIS_IDX = 473  # sağ iris merkezi

class Detector:
    # Başlatma, değişken ayarları, Mediapipe yüz mesh kurulum.
    def __init__(self, user_id):
        self.user_id = user_id
        self.running = True
        self.blink_count = 0
        self.cooldown_counter = 0
        self.frame_counter = 0
        self.ear_values = []
        self.previous_ear = 0.25
        self.blink_threshold = 0.2
        self.consecutive_closed = 0
        self.last_distance_save_time = 0
        self.cooldown_frames = 10
        self.min_blink_frames = 2
        self.calibration_frames = 50

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.gaze_direction_start_time = None
        self.last_gaze_direction = "Center"

        self.head_pose_warning_start = None
        self.warning_displayed = False

        # İlk değerleri orta noktada başlat
        self.prev_avg_x = 0.5
        self.prev_avg_y = 0.5
        self.alpha = 0.2  # smoothing katsayısı (küçük → daha yumuşak)
        self.fixation_durations = []  # her göz açık süresi burada tutulacak
        self.eye_open_start = None
        self.eye_was_open = False

        self.session_start_time = None
        self.session_duration = 0

        # --- Dikkat Dağınıklığı Takip Değişkenleri ---
        self.gaze_off_start = None
        self.head_off_start = None
        self.gaze_loss_logged = False
        self.head_loss_logged = False

        # buffeer periyodik kayıtta bu listeleri kullanacağız
        self._distance_values = []
        self._head_pose_values = []
        self._gaze_values = []
        self._fixation_buffer = []
        #bildirim
        self.last_notification_time = 0
        self.notification_cooldown = 10  # saniye, aynı bildirimi tekrar etmeme aralığı

        self._last_eye_open_log = time.time()
        self._last_fixation_log_time = time.time()


    #Gözün açık/kapalı olduğunu sayısal olarak ölç.
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    #Gözler arası mesafeyi hesapla (ekran uzaklığı için).
    def eye_distance(self, landmarks):
        left = landmarks.part(36)
        right = landmarks.part(45)
        return distance.euclidean((left.x, left.y), (right.x, right.y))
##---------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_eye_landmarks(landmarks, eye_indices, frame_width, frame_height):
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks.landmark[idx].x * frame_width)
            y = int(landmarks.landmark[idx].y * frame_height)
            eye_points.append((x, y))
        return eye_points

    @staticmethod
    def get_iris_point(landmarks, iris_idx, frame_width, frame_height):
        x = int(landmarks.landmark[iris_idx].x * frame_width)
        y = int(landmarks.landmark[iris_idx].y * frame_height)
        return (x, y)

    @staticmethod
    def calculate_norm_iris_pos(iris_point, eye_points):
        xs = [p[0] for p in eye_points]
        ys = [p[1] for p in eye_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        norm_x = (iris_point[0] - min_x) / (max_x - min_x) if max_x != min_x else 0.5
        norm_y = (iris_point[1] - min_y) / (max_y - min_y) if max_y != min_y else 0.5

        return norm_x, norm_y

    def calculate_gaze_direction(self, landmarks, frame_width, frame_height):
        left_eye_points = self.get_eye_landmarks(landmarks, LEFT_EYE_IDX, frame_width, frame_height)
        right_eye_points = self.get_eye_landmarks(landmarks, RIGHT_EYE_IDX, frame_width, frame_height)
        left_iris = self.get_iris_point(landmarks, LEFT_IRIS_IDX, frame_width, frame_height)
        right_iris = self.get_iris_point(landmarks, RIGHT_IRIS_IDX, frame_width, frame_height)

        left_norm_x, left_norm_y = self.calculate_norm_iris_pos(left_iris, left_eye_points)
        right_norm_x, right_norm_y = self.calculate_norm_iris_pos(right_iris, right_eye_points)

        avg_norm_x = (left_norm_x + right_norm_x) / 2
        avg_norm_y = (left_norm_y + right_norm_y) / 2

        # ---- Smoothing (hareketli ortalama) ----
        avg_norm_x = self.alpha * avg_norm_x + (1 - self.alpha) * self.prev_avg_x
        avg_norm_y = self.alpha * avg_norm_y + (1 - self.alpha) * self.prev_avg_y
        self.prev_avg_x, self.prev_avg_y = avg_norm_x, avg_norm_y

        # ---- Sağ/Sol tersini düzelt ----
        avg_norm_x = 1 - avg_norm_x

        if avg_norm_x < 0.45:
            gaze_horizontal = "Right"
        elif avg_norm_x > 0.55:
            gaze_horizontal = "Left"
        else:
            gaze_horizontal = "Center"

        if avg_norm_y < 0.4:
            gaze_vertical = "Up"
        elif avg_norm_y > 0.6  :
            gaze_vertical = "Down"
        else:
            gaze_vertical = "Center"

        gaze_direction = f"{gaze_vertical} - {gaze_horizontal}"

        return {
            "left_norm_x": left_norm_x,
            "left_norm_y": left_norm_y,
            "right_norm_x": right_norm_x,
            "right_norm_y": right_norm_y,
            "avg_norm_x": avg_norm_x,
            "avg_norm_y": avg_norm_y,
            "gaze_direction": gaze_direction,
            "left_eye_points": left_eye_points,
            "right_eye_points": right_eye_points,
            "left_iris": left_iris,
            "right_iris": right_iris
        }

    @staticmethod
    def draw_landmarks(frame, gaze_info):
        # Göz çevresi çizimi
        cv2.polylines(frame, [cv2.convexHull(np.array(gaze_info["left_eye_points"]))], True, (0, 255, 0), 1)
        cv2.polylines(frame, [cv2.convexHull(np.array(gaze_info["right_eye_points"]))], True, (0, 255, 0), 1)

        # İris noktaları
        cv2.circle(frame, gaze_info["left_iris"], 3, (0, 0, 255), -1)
        cv2.circle(frame, gaze_info["right_iris"], 3, (0, 0, 255), -1)
    #---------------------------------------------------------------------------------------------------------------------------
    #Başın pozisyonunu 3D açı olarak hesapla.

    def get_head_pose(self, landmarks, frame):
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Burun ucu
            (0.0, -330.0, -65.0),  # Çene ucu
            (-225.0, 170.0, -135.0),  # Sol göz köşesi
            (225.0, 170.0, -135.0),  # Sağ göz köşesi
            (-150.0, -150.0, -125.0),  # Sol ağız köşesi
            (150.0, -150.0, -125.0)  # Sağ ağız köşesi
        ])


        # --- 2D Noktalar (dlib landmark'larından) ---
        frame_height = frame.shape[0]
        image_points = np.array([
            (landmarks.part(30).x, frame_height - landmarks.part(30).y),
            (landmarks.part(8).x, frame_height - landmarks.part(8).y),
            (landmarks.part(36).x, frame_height - landmarks.part(36).y),
            (landmarks.part(45).x, frame_height - landmarks.part(45).y),
            (landmarks.part(48).x, frame_height - landmarks.part(48).y),
            (landmarks.part(54).x, frame_height - landmarks.part(54).y)
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")


        dist_coeffs = np.zeros((4, 1)) # Lens distortion yok varsayımı

        # --- PnP çözümü ---
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # --- Euler açıları ---
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        def normalize_angle(angle):
            """Açıyı -180 ile +180 aralığına getirir."""
            while angle > 180:
                angle -= 360
            while angle < -180:
                angle += 360
            return angle

        pitch = normalize_angle(float(eulerAngles[0][0]))
        yaw = normalize_angle(float(eulerAngles[1][0]))
        roll = normalize_angle(float(eulerAngles[2][0]))

        # --- Smoothing (titreşimi azaltma) ---
        if not hasattr(self, "prev_pitch"):
            self.prev_pitch, self.prev_yaw, self.prev_roll = pitch, yaw, roll

        alpha = 0.2  # yumuşatma katsayısı
        pitch = alpha * pitch + (1 - alpha) * self.prev_pitch
        yaw = alpha * yaw + (1 - alpha) * self.prev_yaw
        roll = alpha * roll + (1 - alpha) * self.prev_roll

        self.prev_pitch, self.prev_yaw, self.prev_roll = pitch, yaw, roll

        # --- Ekrana yaz ---
        #cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #cv2.putText(frame, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        #print(f"Euler Angles raw: Pitch={eulerAngles[0][0]}, Yaw={eulerAngles[1][0]}, Roll={eulerAngles[2][0]}")

        return pitch, yaw, roll

    # ---------------------------------------------------------------------------------------------------------------------------
    def update_fixation_duration(self, current_ear):
        if current_ear > self.blink_threshold:
            # Göz açık
            if not self.eye_was_open:
                # Yeni açılma başladığında zamanı kaydet
                self.eye_open_start = time.time()
                self.eye_was_open = True
        else:
            # Göz kapalı
            if self.eye_was_open and self.eye_open_start is not None:
                # Açık kalma süresini hesapla
                duration = time.time() - self.eye_open_start
                self.fixation_durations.append(duration)

                # 🔹 Log yoğunluğunu azalt: sadece 5 saniyede bir konsola yaz
                if time.time() - self._last_fixation_log_time >= 5:
                    print(f"Göz Açık Kalma Süresi: {duration:.2f} saniye")
                    self._last_fixation_log_time = time.time()

                self.eye_open_start = None
                self.eye_was_open = False

    #---------------------------------------------------------------------------------------------------------------------------
    #Ana döngü, frame okuyup işlem yapar, tespitleri ekrana yazar ve kaydeder.
    def start(self):
        self.session_start_time = time.time()
        self.running = True

        # Arka planda periyodik kayıt thread’i
        threading.Thread(target=self.start_periodic_save, daemon=True).start()
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            ear = self.previous_ear
            approx_distance = None

            # Gaze hesapla ve yazdır
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                gaze_info = self.calculate_gaze_direction(landmarks, frame_width, frame_height)
                gaze_direction = gaze_info["gaze_direction"]
                self._gaze_values.append(gaze_direction)  # DB için biriktir

                self.draw_landmarks(frame, gaze_info)
            else:
                gaze_direction = "Yok"

            #gaze = self.get_gaze_direction(frame)
            cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)

                # Burada get_head_pose fonksiyonunu çağır
                pitch, yaw, roll = self.get_head_pose(landmarks, frame)
                self._head_pose_values.append((pitch, yaw, roll))  # DB için biriktir

                pitch_threshold = 20
                yaw_threshold = 20
                head_direction = "Ekrana Bakiyor"

                if yaw < -yaw_threshold:
                    head_direction = "Sola Bakiyor"
                elif yaw > yaw_threshold:
                    head_direction = "Saga Bakiyor"

                if pitch > pitch_threshold:
                    head_direction = "Asagi Bakiyor"
                elif pitch < -pitch_threshold:
                    head_direction = "Yukari Bakiyor"

                cv2.putText(frame, f"Kafa Yonu: {head_direction}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)



                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2.0
                self.previous_ear = ear

                # ----------------- Fixation Duration Güncelle -----------------
                self.update_fixation_duration(ear)  # self ile sınıf metodunu çağır

                # Son 5 değerin ortalamasını ekrana yazdır
                if self.fixation_durations:
                    # Ortalama hesaplamadan önce değerleri al
                    avg_fixation = sum(self.fixation_durations[-5:]) / min(5, len(self.fixation_durations))

                    # DB için buffer’a ekle
                    self._fixation_buffer.extend(self.fixation_durations)

                    # Listeyi temizle
                    self.fixation_durations.clear()

                    # Ekrana yazdır
                    cv2.putText(frame, f"Fixation Avg: {avg_fixation:.2f}s", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # --------------------------------------------------------------

                eye_dist_px = self.eye_distance(landmarks)
                approx_distance = 5600 / eye_dist_px
                self._distance_values.append(approx_distance)  # DB için biriktir

                cv2.putText(frame, f"Mesafe: {approx_distance:.1f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if approx_distance <= 30:
                    cv2.putText(frame, "Ekrana COK YAKINSIN!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif approx_distance > 90:
                    cv2.putText(frame, "Ekrandan COK UZAKSIN!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            """  now = time.time()
                if now - self.last_distance_save_time >= 15:
                    save_distance(self.user_id, approx_distance)
                    self.last_distance_save_time = now"""

            if self.frame_counter < self.calibration_frames:
                self.ear_values.append(ear)
                self.frame_counter += 1
                if self.frame_counter == self.calibration_frames:
                    self.blink_threshold = sum(self.ear_values) / len(self.ear_values) * 0.75
                    print(f"Dinamik Blink Esigi: {self.blink_threshold:.3f}")
                continue

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

            if ear < self.blink_threshold:
                self.consecutive_closed += 1
            else:
                if self.consecutive_closed >= self.min_blink_frames and self.cooldown_counter == 0:
                    self.blink_count += 1
                    print(f"Blink Tespit Edildi! Toplam: {self.blink_count}")
                    self.cooldown_counter = self.cooldown_frames
                self.consecutive_closed = 0

            cv2.imshow("Blink & Distance Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            # Kamera kapanınca süreyi hesapla
        if self.session_start_time:
            self.session_duration = time.time() - self.session_start_time
            print(f"Oturum Süresi: {self.session_duration:.2f} saniye")
            save_activity(self.user_id, self.session_duration)  # opsiyonel DB kaydı

        cap.release()
        cv2.destroyAllWindows()
 #---------------------------------------------------------------------------------------------------------------------------
        # 🔸 Windows Bildirim Fonksiyonu

    def show_notification_safe(self,reason):
        """Windows bildirim çubuğuna uyarı gönderir"""
        try:
            notification.notify(
                title="⚠️ Dikkat Dağınıklığı Algılandı!",
                message="Ekrandan uzaklaştınız veya başınızı çevirdiniz.",
                app_name="Dikkat Takip Sistemi",
                timeout=6  # saniye
            )
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🟡 Windows bildirimi gönderildi.")
        except Exception as e:
            print(f"Bildirim hatası: {e}")
#---------------------------------------------------------------------------------------------------------------------------
    #Döngüyü durdurur.
    def stop(self):
        self.running = False
    #Arka planda belirli aralıklarla blink verisini veritabanına kaydeder.    # Arka planda belirli aralıklarla verileri veritabanına kaydeder.
    def start_periodic_save(self):

        def save_loop():
            blink_interval = 10
            distance_interval = 15
            head_pose_interval = 30
            gaze_interval = 30
            fixation_interval = 30
            distraction_check_interval = 2  # her 2 saniyede bir kontrol et
            distraction_threshold = 10

            last_blink_time = time.time()
            last_distance_time = time.time()
            last_head_pose_time = time.time()
            last_gaze_time = time.time()
            last_fixation_time = time.time()
            last_distraction_check = time.time()

            # Dikkat takibi için değişkenler
            gaze_off_start = None
            distraction_logged = False  # aynı dikkatsizliği bir kez kaydetmek için

            while self.running:
                now = time.time()

                # Blink → 10s toplam
                if now - last_blink_time >= blink_interval:
                    if self.blink_count > 0:
                        save_blink(self.user_id, self.blink_count)
                        print(f"DB → Blink: {self.blink_count}")
                        self.blink_count = 0  # sadece burada sıfırla
                    last_blink_time = now

                # Mesafe → 15s ortalama
                if now - last_distance_time >= distance_interval:
                    if self._distance_values:
                        # Listeyi atomik sayılabilecek şekilde kopyalayıp temizleyelim
                        vals = self._distance_values[:]
                        self._distance_values.clear()
                        avg_distance = sum(vals) / len(vals)
                        save_distance(self.user_id, avg_distance)
                        print(f"DB → Mesafe: {avg_distance:.2f} cm")
                    last_distance_time = now

                # Baş yönü → 30s ortalama yaw/pitch/roll
                if now - last_head_pose_time >= head_pose_interval:
                    if self._head_pose_values:
                        vals = self._head_pose_values[:]
                        self._head_pose_values.clear()
                        avg_pitch = sum(p for p, _, _ in vals) / len(vals)
                        avg_yaw = sum(y for _, y, _ in vals) / len(vals)
                        avg_roll = sum(r for _, _, r in vals) / len(vals)
                        save_head_position(self.user_id, avg_pitch, avg_yaw, avg_roll)
                        print(f"DB → Head Pose: pitch={avg_pitch:.2f}, yaw={avg_yaw:.2f}, roll={avg_roll:.2f}")
                    last_head_pose_time = now

                # Göz yönü → 30s ekran bakış yüzdesi
                if now - last_gaze_time >= gaze_interval:
                    if self._gaze_values:
                        vals = self._gaze_values[:]
                        self._gaze_values.clear()
                        # 'Center - Center' sayısı / toplam
                        screen_percentage = (vals.count("Center - Center") / len(vals)) * 100.0
                        save_eye_direction(self.user_id, screen_percentage)
                        print(f"DB → Gaze Screen %: {screen_percentage:.2f}%")
                    last_gaze_time = now

                # Fixation → 30s ortalama fixation süresi
                if now - last_fixation_time >= fixation_interval:
                    if self._fixation_buffer:
                        vals = self._fixation_buffer[:]
                        self._fixation_buffer.clear()
                        avg_fixation = sum(vals) / len(vals)
                        save_eye_open_time(self.user_id, avg_fixation)
                        print(f"DB → Fixation: {avg_fixation:.2f} s")
                    last_fixation_time = now

                    # 🔹 Dikkat dağınıklığı kontrolü (her 2 saniyede bir)
                if now - last_distraction_check >= distraction_check_interval:
                    # Son bilinen göz ve baş yönlerini al
                    current_gaze = self._gaze_values[-1] if self._gaze_values else "Yok"
                    current_head = self._head_pose_values[-1] if self._head_pose_values else (0, 0, 0)

                    avg_distance = None
                    if self._distance_values:
                        avg_distance = sum(self._distance_values[-5:]) / min(len(self._distance_values), 5)

                    # Göz ekrandan uzak mı?
                    gaze_off = current_gaze != "Center - Center"
                    # Baş yönü çok mu sapmış?
                    _, yaw, _ = current_head
                    head_off = abs(yaw) > 25  # eşik 25 derece

                    too_far = avg_distance is not None and avg_distance > 80  # çok uzak
                    too_close = avg_distance is not None and avg_distance < 30  # çok yakın
                    low_blink = self.blink_count < 1  # 10 saniyede hiç göz kırpma yoksa
                    high_blink = self.blink_count > 8  # 10 saniyede 8’den fazla blink varsa

                    reason = None

                    # Hangi neden varsa belirle
                    if too_far:
                        reason = "Ekrandan çok uzaklaştın."
                    elif too_close:
                        reason = "Ekrana çok yaklaştın."
                    elif gaze_off:
                        reason = "Ekrana bakmıyorsun."
                    elif head_off:
                        reason = "Başını çevirdin."
                    elif low_blink:
                        reason = "Uzun süredir göz kırpmadın."
                    elif high_blink:
                        reason = "Çok sık göz kırpıyorsun."

                        # Neden varsa dikkatsizlik say
                    if reason:
                        if gaze_off_start is None:
                            gaze_off_start = now
                        elif now - gaze_off_start >= distraction_threshold and not distraction_logged:
                            print(f"⚠️ Dikkat Dağınıklığı Algılandı! ({reason})")
                            save_distraction_event(self.user_id, now, reason)
                            self.show_notification_safe(reason)
                            distraction_logged = True
                    else:
                        gaze_off_start = None
                        distraction_logged = False

                    last_distraction_check = now
                time.sleep(0.5)

        Thread(target=save_loop, daemon=True).start()