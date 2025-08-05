from pickle import FRAME
import cv2
import sys
from cv2.gapi.streaming import desync
import pandas as pd
import numpy as np
from pandas._config import display
from pandas.core.internals.array_manager import NullArrayProxy
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import json

# input args
if len(sys.argv) != 3:
    print("Usage: tracker.py <video_file> <input times>")
    sys.exit(1)

cap = cv2.VideoCapture(sys.argv[1])
tfrs = pd.read_csv(sys.argv[2])

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read()
ret, frame2 = cap.read()

tr_pts = []
tr_time_pts = []

print(frame1.shape)

while cap.isOpened():
    if not ret:
        break

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contour

    time_stamp = np.round(cap.get(cv2.CAP_PROP_POS_MSEC))

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        c_x = int(x + w/2)
        c_y = int(y + h/2)

        if cv2.contourArea(contour) < 500:
            continue

        if not (c_x <= 150 and c_y <= 150):
            print(c_x, c_y)
            tr_pts.extend([(c_x, c_y)])
            tr_time_pts.extend([{"Ts": time_stamp, "X": c_x, "Y": c_y}])

            nd_tr = np.array(tr_pts[-100:])
            rv = (int(c_x), int(c_y))

            cv2.polylines(frame1, [nd_tr], False, (255, 0, 0), 3)
            cv2.circle(frame1, rv, 15, (0, 0, 255), 2)

    image = cv2.resize(frame1, (1280,720))
    frame1 = frame2
    ret, frame2 = cap.read()

    try:
        rv_diff = cv2.absdiff(frame1, frame2)
        rv_diff_res = rv_diff.astype(np.uint8)
        rv_percentage = (np.count_nonzero(rv_diff_res) * 100) / rv_diff_res.size

        print("rv:", rv_percentage)
        if rv_percentage > 15:
            print("Unstable frame!")
            frame1 = frame2
    except Exception as e:
        print("e final")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

ts_out = []
for index, row in tfrs.iterrows():
    ts = row[0]
    vrow = json.loads(json.dumps(min(tr_time_pts, key=lambda x:abs(x["Ts"]-ts))))
    vrow["Ts"] = ts
    ts_out.append(vrow)

pd.DataFrame(ts_out).to_csv("output.csv", index=False)
