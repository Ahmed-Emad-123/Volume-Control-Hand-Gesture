import cv2 as cv
import numpy as np
import math
from HandDetect import HandDetection
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------------- Using Hand Detection Module ---------------- #
detector_hand = HandDetection()

# ---------------- Activate volume with PC ---------------- #
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# volume.GetMute() == 0
# volume.GetMasterVolumeLevel() == -20db
volume_range = volume.GetVolumeRange()         ## range from max vol. (0) and min vol. (-65.25)

# ---------------- Set Volume Range  ---------------- #
min_vol = volume_range[0]
max_vol = volume_range[1]

# ---------------- Volume Parameters  ---------------- #
vol = 0
vol_bar = 400
vol_perc = 0

# ---------------- Display Camera  ---------------- #
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret is not True:
        break

    # ---------------- Declare Hand Detect Module For Detection ---------------- #
    frame = detector_hand.hand_detect(frame)
    landmark_list = detector_hand.find_position(frame)

    if landmark_list is not None and len(landmark_list) != 0:
        # print(landmark_list[4])

        # ---------------- Access the coordinate of hand's id num 4 and 8 ---------------- #
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]

        # ---------------- Getting center of the line ---------------- #
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # ---------------- Drawing a circle and line on them to be noticeable for observer ---------------- #
        cv.circle(frame, (x1, y1), color=(255, 0, 0), thickness=-1, radius=10)
        cv.circle(frame, (x2, y2), color=(255, 0, 0), thickness=-1, radius=10)
        cv.line(frame, (x1, y1), (x2, y2), color=(0,0,255), thickness=3)
        cv.circle(frame, (cx, cy), thickness=-1, radius=10, color=(255, 0, 0))

        # ---------------- Getting length of the line using math ---------------- #
        length = math.hypot(x2 - x1, y2 - y1)

        ######## Hand range is from 264 to 50 ########
        ######## Volume range is from -65 to 0 ########

        # ---------------- Changing hand range to volume range using numpy (getting hand range equivalent to volume range) ---------------- #
        vol = np.interp(length, [50, 264], [min_vol, max_vol])
        vol_bar = np.interp(length, [50, 264], [400, 150])
        vol_perc = np.interp(length, [50, 264], [0, 100])

        # ---------------- Master volume of your computer audio ---------------- #
        volume.SetMasterVolumeLevel(vol, None)

        if length <= 50:
            cv.circle(frame, (cx, cy), thickness=-1, radius=10, color=(0, 255, 0))

    # ---------------- Drawing rectangle that represent the volume bar in the frame ---------------- #
    cv.rectangle(frame, (50, 150), (85, 400), (0,255,0), 3)
    cv.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)
    cv.putText(frame, text=f'{int(vol_perc)} %', org=(40, 450), color=(255, 0, 0), fontFace=cv.FONT_HERSHEY_PLAIN, thickness=3, fontScale=3)

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyWindow()

"""
interpolation mean is try to fit or connect known data with other data

"""

