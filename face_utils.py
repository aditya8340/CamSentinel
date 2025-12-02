# face_utils.py
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import face_recognition

def pil_load_and_fix(path, max_dim=1600):
    """
    Open with PIL, convert to RGB, optionally resize large images,
    return a uint8 contiguous RGB numpy array.
    """
    img = Image.open(path)
    img = img.convert("RGB")  # drops alpha, converts CMYK etc
    if max_dim:
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
    arr = np.array(img)  # RGB, dtype usually uint8
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)

def sanitize_array_to_rgb_uint8(arr, from_bgr=False, max_dim=1600):
    """
    Given a numpy image array (maybe BGR as from cv2), return
    RGB uint8 contiguous array. If array is weird, use PIL fallback.
    """
    if arr is None:
        raise ValueError("Received None image array in sanitizer")

    # If float image in 0..1
    if np.issubdtype(arr.dtype, np.floating):
        arr = (255 * np.clip(arr, 0, 1)).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # If grayscale -> convert to RGB
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # has alpha; convert via PIL to be safe
        pil = Image.fromarray(arr)
        pil = pil.convert("RGB")
        arr = np.array(pil)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        if from_bgr:
            # convert from BGR->RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        # else assume it's already RGB
    else:
        # fallback through PIL to be safe (handles odd formats)
        pil = Image.fromarray(arr)
        pil = pil.convert("RGB")
        arr = np.array(pil)

    # Resize if huge
    if max_dim is not None:
        h, w = arr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(arr)

def safe_face_encodings_from_array(arr, from_bgr=False):
    """
    Return face encodings (list) from a numpy array after sanitizing.
    Catches and attempts fallback if dlib/face_recognition raises errors.
    """
    try:
        arr_fixed = sanitize_array_to_rgb_uint8(arr, from_bgr=from_bgr)
        return face_recognition.face_encodings(arr_fixed)
    except Exception as e:
        # Try a fallback: write to a temp JPEG and reload via face_recognition.load_image_file
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()
            # If arr likely BGR (from cv2), convert to BGR for cv2.imwrite
            to_write = arr
            if not from_bgr and np.array_equal(arr_fixed[:, :, ::-1], arr):
                # arr was RGB, cv2.imwrite expects BGR -> convert
                to_write = cv2.cvtColor(arr_fixed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp_path, to_write)
            loaded = face_recognition.load_image_file(tmp_path)
            encs = face_recognition.face_encodings(loaded)
            os.remove(tmp_path)
            return encs
        except Exception as e2:
            print("Fallback face encoding failed:", e2)
            return []
