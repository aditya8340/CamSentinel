import face_recognition
import cv2
import numpy as np

# Load the image using OpenCV
image_path = "test_face.jpg"  # or your actual image path
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Could not load image from {image_path}")

# Convert to RGB (OpenCV loads BGR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

print("Loaded image shape:", image_rgb.shape, "dtype:", image_rgb.dtype)

# Ensure dtype is uint8 and contiguous
image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

# Detect face locations
try:
    face_locations = face_recognition.face_locations(image_rgb, model="hog")
    print(f"Detected {len(face_locations)} faces.")
except Exception as e:
    print("face_recognition error:", str(e))

# Draw rectangles around detected faces
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show result
cv2.imshow("Detected Faces", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
