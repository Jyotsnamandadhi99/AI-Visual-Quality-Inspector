"""
Processed video generation and automatic output saving

Author: Jyotsna Mandadhi
"""

import cv2

video_path = r"F:\INNOMATICS\ML\Open_Cv\AI-Visual-Quality-Inspector\data\raw\sample.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

image_count = 0
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # GrayScale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(
        gray,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,   
        2
    )

    cv2.imshow("Gray Video", gray)

    # Edges
    #edges = cv2.Canny(gray,100,200)
    #cv2.imshow("Edges", edges)

    # Frame BGR video
    #cv2.imshow("Video", frame)

    key = cv2.waitKey(25) & 0xFF
    if key == ord('s'):
        filename = f"outputs/Screen/video_capture{image_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot Saved: {filename}")
        image_count += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

'''out = cv2.VideoWriter(
    "outputs/Videos/Frame.mp4",
    fourcc,
    fps,
    (width, height),
    False
)'''
cap.release()
cv2.destroyAllWindows()

