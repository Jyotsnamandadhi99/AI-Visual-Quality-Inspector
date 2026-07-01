import cv2
image = cv2.imread(r"F:\INNOMATICS\ML\Open_Cv\AI-Visual-Quality-Inspector\data\raw\sample.jfif")
if image is None:
    print("Image not found!")
else:
    print("Image Shape:",image.shape)
    cv2.imshow("Sample Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
