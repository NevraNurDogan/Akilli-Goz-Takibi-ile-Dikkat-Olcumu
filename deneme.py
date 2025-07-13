import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera açılamadı.")
else:
    print("✅ Kamera açıldı.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🔴 Görüntü alınamadı.")
            break

        cv2.imshow("Kamera Testi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
