import cv2
from cv2.gapi.streaming import desync
import pandas as pd
import numpy as np
from pandas._config import display
from pandas.core.internals.array_manager import NullArrayProxy
from super_image import EdsrModel, ImageLoader
from PIL import Image
from pathlib import Path

# tunable params
bounding_w = 72
bounding_h = 72
scale_factor = 4

# initialize temp dirs
Path("./training_images").mkdir(exist_ok=True)

# load models and input data
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=scale_factor)
gt_pts = pd.read_csv('groundtruth.csv')

# video stream setup
cap = cv2.VideoCapture('practice video.mp4')

if not cap.isOpened():
    print("Error opening video file")

# state loop
frame_no = 0
ipt_num = 0
disp_pts = []

while cap.isOpened():
    # video frame input and time sync
    ret, frame = cap.read()
    frame = np.array(frame)
    if not ret:
        break

    time_stamp = np.round(cap.get(cv2.CAP_PROP_POS_MSEC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    pt_df = list(gt_pts[gt_pts['TIMESTAMP'] == time_stamp].drop('TIMESTAMP', axis=1).astype(int).itertuples(index=False, name=None))

    # process and display point
    disp_pts.extend(pt_df)

    if pt_df.__len__() != 0:
        hi, hf = int(pt_df[0][1]-(bounding_h/2)), int(pt_df[0][1]+(bounding_h/2))
        wi, wf = int(pt_df[0][0]-(bounding_w/2)), int(pt_df[0][0]+(bounding_w/2))

        if (not any([v < 0 or v > height for v in [hi, hf]])) and (not any([v < 0 or v > width for v in [wi, wf]])):
            pil_cut = Image.fromarray(cv2.cvtColor(frame[hi:hf, wi:wf], cv2.COLOR_BGR2RGB))
            fc_inputs = ImageLoader.load_image(pil_cut)

            upscaled_cut = model(fc_inputs)
            ImageLoader.save_image(upscaled_cut, './training_images/{}.png'.format(str(int(time_stamp))))

            ipt_num += 1

        nd_r = np.array(disp_pts[-100:])

        cv2.polylines(frame, [nd_r], False, (0, 0, 255), 3)
        cv2.circle(frame, pt_df[0], 15, (255, 0, 0), 2)

    # display the frame
    cv2.putText(frame, "timestamp: " + str(time_stamp), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("DRL Pfizer Challenge", frame)

    # keyboard controls
    key = cv2.waitKey(1)

    if key == ord('p'):
        cv2.waitKey(-1)
    elif key == ord('q'):
        break

    frame_no += 1

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

print("Found and captured:", ipt_num, "frames.")
