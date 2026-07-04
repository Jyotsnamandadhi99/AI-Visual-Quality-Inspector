import cv2

video_path = r"F:\INNOMATICS\ML\Open_Cv\AI-Visual-Quality-Inspector\data\raw\sample.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

gray_mode = False
while True:
    ret, frame = cap.read()
    if not ret:
        break


    # GrayScale
    if gray_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Video", gray)
    else:
        cv2.imshow("Video", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('g'):
        gray_mode = not gray_mode

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

