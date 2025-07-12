import cv2
import dlib
import time
from scipy.spatial import distance
from threading import Thread
from database import save_blink, save_distance

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

class Detector:
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

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def eye_distance(self, landmarks):
        left = landmarks.part(36)
        right = landmarks.part(45)
        return distance.euclidean((left.x, left.y), (right.x, right.y))

    def start(self):
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            ear = self.previous_ear
            approx_distance = None

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
                ear_left = self.eye_aspect_ratio(left_eye)
                ear_right = self.eye_aspect_ratio(right_eye)
                ear = (ear_left + ear_right) / 2.0
                self.previous_ear = ear

                eye_dist_px = self.eye_distance(landmarks)
                approx_distance = 5600 / eye_dist_px
                cv2.putText(frame, f"Mesafe: {approx_distance:.1f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if approx_distance <= 40:
                    cv2.putText(frame, "Ekrana COK YAKINSIN!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif approx_distance > 80:
                    cv2.putText(frame, "Ekrandan COK UZAKSIN!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                now = time.time()
                if now - self.last_distance_save_time >= 15:
                    save_distance(self.user_id, approx_distance)
                    self.last_distance_save_time = now

            if self.frame_counter < self.calibration_frames:
                self.ear_values.append(ear)
                self.frame_counter += 1
                if self.frame_counter == self.calibration_frames:
                    self.blink_threshold = sum(self.ear_values) / len(self.ear_values) * 0.75
                    print(f"Dinamik Blink Eşiği: {self.blink_threshold:.3f}")
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

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

    def start_periodic_save(self):
        def save_loop():
            while self.running:
                time.sleep(10)
                if self.blink_count > 0:
                    save_blink(self.user_id, self.blink_count)
                    print(f"Veritabanına kaydedildi: {self.blink_count} blink")
                    self.blink_count = 0
        Thread(target=save_loop, daemon=True).start()
