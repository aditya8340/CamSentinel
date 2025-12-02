import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Configuration ---
PROJECT_DIR = r"D:\FaceTrackingProject"
MAIN_SCRIPT = os.path.join(PROJECT_DIR, "yolo_deepsort_facenet_opt.py")

# --- Helper to run detection ---
def run_detection(video_input, preview_enabled):
    """Launch the main detection script with the chosen video input."""
    if not os.path.exists(MAIN_SCRIPT):
        messagebox.showerror("Error", f"Main script not found:\n{MAIN_SCRIPT}")
        return

    try:
        preview_arg = "--preview" if preview_enabled else "--no-preview"
        cmd = ["python", MAIN_SCRIPT, str(video_input), preview_arg]
        subprocess.Popen(cmd, cwd=PROJECT_DIR)
        messagebox.showinfo("CamSentinel", "Detection started. Check console for progress.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run detection:\n{e}")

# --- Actions ---
def choose_camera():
    run_detection(0, preview_var.get())

def choose_video():
    filepath = filedialog.askopenfilename(
        initialdir=PROJECT_DIR,
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")]
    )
    if filepath:
        run_detection(filepath, preview_var.get())
    else:
        messagebox.showinfo("CamSentinel", "No video selected.")

# --- Tkinter UI setup ---
root = tk.Tk()
root.title("CamSentinel - Smart CCTV Frontend")
root.geometry("460x310")
root.configure(bg="#101820")

# Title
title = tk.Label(root, text="CamSentinel", font=("Segoe UI", 24, "bold"),
                 fg="#00ADEF", bg="#101820")
title.pack(pady=20)

subtitle = tk.Label(root, text="Smart CCTV Monitoring Interface",
                    font=("Segoe UI", 11), fg="#AAAAAA", bg="#101820")
subtitle.pack(pady=(0, 20))

# Checkbox for preview toggle
preview_var = tk.BooleanVar(value=False)
preview_checkbox = tk.Checkbutton(
    root,
    text="Enable Live Preview (Slower)",
    variable=preview_var,
    fg="white",
    bg="#101820",
    selectcolor="#101820",
    activebackground="#101820",
    font=("Segoe UI", 10)
)
preview_checkbox.pack(pady=5)

# Buttons
btn_camera = tk.Button(
    root, text="Switch to Camera",
    font=("Segoe UI", 12, "bold"),
    bg="#00ADEF", fg="white",
    width=20, height=2, relief="flat",
    command=choose_camera
)
btn_camera.pack(pady=10)

btn_video = tk.Button(
    root, text="Upload / Choose Video",
    font=("Segoe UI", 12, "bold"),
    bg="#007ACC", fg="white",
    width=20, height=2, relief="flat",
    command=choose_video
)
btn_video.pack(pady=10)

# Footer
footer = tk.Label(root,
    text="Developed by Aditya Yadav | AI & ML | v1.0",
    font=("Segoe UI", 9), fg="#777777", bg="#101820")
footer.pack(side="bottom", pady=10)

root.mainloop()
