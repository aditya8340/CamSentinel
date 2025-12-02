import os
import cv2
import numpy as np
import face_recognition

db_path = "faces_db_converted"  # updated folder
print("Loading faces from:", db_path)

def safe_load_and_encode(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to load {img_path}")
            return []

        # Convert to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ✅ Fix alignment: ensure C-contiguous and memory aligned
        rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)

        # ✅ Additional fix: force data alignment in new array
        rgb_img = np.require(rgb_img, dtype=np.uint8, requirements=["C", "OWNDATA"])

        # Try face detection
        encodings = face_recognition.face_encodings(rgb_img)

        return encodings

    except Exception as e:
        print(f"face_recognition error: {e}")
        # Try fallback with PIL
        try:
            from PIL import Image
            img_pil = Image.open(img_path).convert("RGB")
            rgb_img = np.array(img_pil)
            rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)
            return face_recognition.face_encodings(rgb_img)
        except Exception as e2:
            print(f"Fallback face encoding failed for {img_path}, {e2}")
            return []

# Iterate through database
people_encodings = {}
for person in os.listdir(db_path):
    person_folder = os.path.join(db_path, person)
    if not os.path.isdir(person_folder):
        continue

    encodings = []
    for img_file in os.listdir(person_folder):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_folder, img_file)
            enc = safe_load_and_encode(img_path)
            print(f"{img_path} -> encodings: {len(enc)}")
            if len(enc) > 0:
                encodings.extend(enc)

    if len(encodings) > 0:
        people_encodings[person] = encodings
        print(f"✅ Loaded encodings for {person}")
    else:
        print(f"⚠️ No valid encodings found for {person}")

print(f"\n✅ Loaded face encodings for {len(people_encodings)} people.")
