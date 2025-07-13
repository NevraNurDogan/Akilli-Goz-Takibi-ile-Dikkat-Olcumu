import logging
import os
import cv2
import dlib
import time
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from threading import Thread
from database import save_blink, save_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

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

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]
        self.gaze_direction_start_time = None
        self.last_gaze_direction = "Center"

        self.head_pose_warning_start = None
        self.warning_displayed = False

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def eye_distance(self, landmarks):
        left = landmarks.part(36)
        right = landmarks.part(45)
        return distance.euclidean((left.x, left.y), (right.x, right.y))

    def get_gaze_direction(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_iris = landmarks[468]
        left_eye_outer = landmarks[33]
        left_eye_inner = landmarks[133]
        right_iris = landmarks[473]
        right_eye_outer = landmarks[362]
        right_eye_inner = landmarks[263]

        def get_ratio(iris, eye_inner, eye_outer):
            iris_x = iris.x * w
            inner_x = eye_inner.x * w
            outer_x = eye_outer.x * w
            eye_width = abs(inner_x - outer_x)
            ratio = (iris_x - outer_x) / eye_width
            return ratio

        left_ratio = get_ratio(left_iris, left_eye_inner, left_eye_outer)
        right_ratio = get_ratio(right_iris, right_eye_inner, right_eye_outer)
        avg_ratio = (left_ratio + right_ratio) / 2

        if avg_ratio < 0.35:
            return "Right"
        elif avg_ratio > 0.65:
            return "Left"
        else:
            return "Center"

    def get_head_pose(self, landmarks, frame):
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),
            (landmarks.part(8).x, landmarks.part(8).y),
            (landmarks.part(36).x, landmarks.part(36).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(31).x, landmarks.part(31).y),
            (landmarks.part(35).x, landmarks.part(35).y)
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                    dist_coeffs)

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        def normalize_angle(angle):
            angle = angle % 360
            if angle > 180:
                angle -= 360
            return angle

        pitch = normalize_angle(float(eulerAngles[0][0]))
        yaw = normalize_angle(float(eulerAngles[1][0]))
        roll = normalize_angle(float(eulerAngles[2][0]))

        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return pitch, yaw, roll
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

            gaze = self.get_gaze_direction(frame)
            cv2.putText(frame, f"Gaze: {gaze}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if faces:
                face = faces[0]
                landmarks = predictor(gray, face)
                pitch, yaw, roll = self.get_head_pose(landmarks, frame)

                # Basit yorumlama
                pitch_threshold = 15
                yaw_threshold = 15
                head_direction = "Ekraya Bakiyor"

                if yaw < -yaw_threshold:
                    head_direction = "Sola Bakiyor"
                elif yaw > yaw_threshold:
                    head_direction = "Saga Bakiyor"

                if pitch > pitch_threshold:
                    head_direction = "Aşagi Bakiyor"
                elif pitch < -pitch_threshold:
                    head_direction = "Yukari Bakiyor"

                # Ekranda yazdır
                cv2.putText(frame, f"Kafa Yönü: {head_direction}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
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