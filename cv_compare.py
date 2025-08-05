import cv2
import sys
from cv2.gapi.streaming import desync
import pandas as pd
import numpy as np
from pandas._config import display
from pandas.core.internals.array_manager import NullArrayProxy
from super_image import EdsrModel, ImageLoader
from PIL import Image
from pathlib import Path

# input args
if len(sys.argv) != 3:
    print("Usage: track.py <video_file> <input times>")
    sys.exit(1)

cap = cv2.VideoCapture(sys.argv[1]) # Open the Video

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = cap.read() # We define two frame one after another
ret, frame2 = cap.read()

gt_pts = pd.read_csv('groundtruth.csv')
disp_pts = []
tr_pts = []

print(frame1.shape)
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) # To find out absolute difference of first frame and second frame
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # convert it to gray scale - We do it for contour stages (It is easier to find contour with gray scale)
    blur = cv2.GaussianBlur(gray, (5,5), 0) # Blur the grayscale frame
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY) # max threshold values is 255 - we need trashold value
    dilated = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contour
    time_stamp = np.round(cap.get(cv2.CAP_PROP_POS_MSEC))

    pt_df = list(gt_pts[gt_pts['TIMESTAMP'] == time_stamp].drop('TIMESTAMP', axis=1).astype(int).itertuples(index=False, name=None))
    disp_pts.extend(pt_df)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        c_x = int(x + w/2)
        c_y = int(y + h/2)

        if cv2.contourArea(contour) < 500:
            continue

        if pt_df.__len__() != 0 and not (c_x <= 150 and c_y <= 150):
            print(c_x, c_y)
            tr_pts.extend([(c_x, c_y)])

            nd_r = np.array(disp_pts[-100:])
            nd_tr = np.array(tr_pts[-100:])
            rv = (int(c_x), int(c_y))

            cv2.polylines(frame1, [nd_r], False, (0, 0, 255), 3)
            cv2.polylines(frame1, [nd_tr], False, (255, 0, 0), 3)

            cv2.circle(frame1, pt_df[0], 15, (0, 0, 255), 2)
            #cv2.circle(frame1, rv, 15, (255, 0, 0), 2)

    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (1280,720))
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
