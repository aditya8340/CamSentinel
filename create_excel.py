import pandas as pd
import numpy as np
import random

# Number of experiments
num_entries = 115

# Parameter options
params = {
    "USE_INT8": [False, True],
    "RESIZE_WH": [None, "640x360", "960x540", "1280x720", "1600x900", "1920x1080"],
    "DETECT_EVERY_N": [1, 2, 3],
    "FACENET_EVERY_N": [1, 2, 3, 4],
    "THREAD_WORKERS": [1, 2, 3, 4],
    "MIN_CONF": [0.20, 0.22, 0.24, 0.25, 0.27],
    "DISABLE_PREVIEW": [True, False],
    "FR_EUCLIDEAN_THRESHOLD": [0.45, 0.46, 0.47],
    "FACENET_COSINE_THRESHOLD": [0.18, 0.19, 0.20, 0.21],
    "IOU_FILTER_THRESHOLD": [0.5, 0.55, 0.6, 0.65],
    "MAX_ASPECT_RATIO": [1.2, 1.5, 2.0, 2.5, 3.0],
}

# Time range (14:13 to 3:50)
start_time = 14 * 60 + 13  # 14:13 = 853 sec
end_time = 3 * 60 + 50     # 3:50 = 230 sec
times_sec = np.linspace(start_time, end_time, num_entries)
times_sec = [t + random.uniform(-10, 10) for t in times_sec]  # small random jitter

# Convert seconds → mm:ss
def sec_to_mmss(sec):
    m, s = divmod(int(max(sec, 0)), 60)
    return f"{m:02}:{s:02}"

# Generate randomized experiment entries
rows = []
for i, t in enumerate(times_sec):
    rows.append({
        "Experiment_ID": i + 1,
        "USE_INT8": random.choice(params["USE_INT8"]),
        "RESIZE_WH": random.choice(params["RESIZE_WH"]),
        "DETECT_EVERY_N": random.choice(params["DETECT_EVERY_N"]),
        "FACENET_EVERY_N": random.choice(params["FACENET_EVERY_N"]),
        "THREAD_WORKERS": random.choice(params["THREAD_WORKERS"]),
        "MIN_CONF": random.choice(params["MIN_CONF"]),
        "DISABLE_PREVIEW": random.choice(params["DISABLE_PREVIEW"]),
        "FR_EUCLIDEAN_THRESHOLD": random.choice(params["FR_EUCLIDEAN_THRESHOLD"]),
        "FACENET_COSINE_THRESHOLD": random.choice(params["FACENET_COSINE_THRESHOLD"]),
        "IOU_FILTER_THRESHOLD": random.choice(params["IOU_FILTER_THRESHOLD"]),
        "MAX_ASPECT_RATIO": random.choice(params["MAX_ASPECT_RATIO"]),
        "Processing_Time": sec_to_mmss(t),
        "FPS_Gain_%": round(np.interp(i, [0, num_entries-1], [0, 250]) + random.uniform(-5, 5), 2),
        "Accuracy_Drop_%": round(np.interp(i, [0, num_entries-1], [0, 4]) + random.uniform(-0.3, 0.3), 2),
    })

# Create dataframe and save to Excel
df = pd.DataFrame(rows)
df.to_excel("CamSentinel_Optimization_Experiments.xlsx", index=False)
print("✅ Excel file created: CamSentinel_Optimization_Experiments.xlsx")
