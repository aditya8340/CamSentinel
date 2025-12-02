import os
import cv2
import numpy as np
from PIL import Image

input_folder = "faces_db/Aman"
output_folder = "faces_db_converted/Aman"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    in_path = os.path.join(input_folder, filename)
    out_path = os.path.join(output_folder, filename)

    img = cv2.imread(in_path)
    if img is None:
        print(f"⚠️ Could not read {in_path}")
        continue

    # ✅ Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ Ensure 8-bit RGB with contiguous memory
    rgb_img = np.ascontiguousarray(rgb_img, dtype=np.uint8)

    # ✅ Save cleanly using Pillow to ensure proper file encoding
    Image.fromarray(rgb_img).save(out_path, format="JPEG", quality=95)

    print(f"✅ Converted and saved: {out_path}")

print("\nAll images converted and saved to:", output_folder)
