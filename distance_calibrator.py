import cv2
import dlib
from scipy.spatial import distance

# Gözler arası piksel mesafesini hesapla
def eye_distance(landmarks):
    left = landmarks.part(36)
    right = landmarks.part(45)
    return distance.euclidean((left.x, left.y), (right.x, right.y))

# Kalibrasyon fonksiyonu
def calibrate_distance(real_cm):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)

    print("Lütfen kameraya bak ve yüzünü ortala...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Gözler arası mesafe (piksel)
            px_dist = eye_distance(landmarks)
            calibration_constant = real_cm * px_dist

            cv2.putText(frame, f"Px Mesafe: {px_dist:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Sabit (cm * px): {calibration_constant:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            print(f"\nGerçek Mesafe: {real_cm} cm")
            print(f"Gözler Arası Piksel Mesafesi: {px_dist:.2f}")
            print(f"Kalibrasyon Sabiti (cm * px): {calibration_constant:.2f}")

            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow("Kalibrasyon - Kameraya Bakin", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        real_cm = float(input("Kameradan kaç cm uzaktasın? (örn: 50): "))
        calibrate_distance(real_cm)
    except ValueError:
        print("Geçerli bir sayı girilmedi.")
