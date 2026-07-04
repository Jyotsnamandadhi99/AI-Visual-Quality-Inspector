import cv2

cap = cv2.VideoCapture(0)

image_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    mirrored = cv2.flip(frame, 1)

    cv2.imshow("Webcam", mirrored)
    #cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"outputs/Captured_images/captured_{image_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        image_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()