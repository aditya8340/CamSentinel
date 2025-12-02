import cv2

image = cv2.imread("test_face.jpg")
print("Image loaded:", image is not None)

# Resize to fit the screen (optional)
resized = cv2.resize(image, (800, 600))  # You can adjust these values
cv2.imshow("Test", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
