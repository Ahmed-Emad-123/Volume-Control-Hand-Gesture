import cv2 as cv
# from mediapipe.python import *
import mediapipe.python.solutions as solutions
import mediapipe.python.solutions.drawing_utils

class HandDetection():

    # ---------------- Attributes of Hands() function for controlling detection ---------------- #
    def __init__(self, mode=False, max_hand=2, detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hand = max_hand
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        # ---------------- Using mediapipe hand functions for detection ---------------- #
        self.mp_hand = mediapipe.python.solutions.hands
        self.hands = self.mp_hand.Hands()
        self.mp_draw = mediapipe.python.solutions.drawing_utils

    def hand_detect(self, frame, draw = True):
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.result = self.hands.process(image_rgb)

        # ---------------- Detecting the landmarks using multi_hand_landmarks for hand ---------------- #
        if self.result.multi_hand_landmarks:
            for hand_lm in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(frame, hand_lm, self.mp_hand.HAND_CONNECTIONS)

        return frame

    # ---------------- Getting position of hand using id and landmark ---------------- #
    def find_position(self, frame, hand_num = 0, draw = True):
        landmark_list = []
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[hand_num]    # hand_num = number of hands in the cam

            # ---------------- Rendering Landmarks ---------------- #
            for id, landmk in enumerate(my_hand.landmark):
                h, w, c = frame.shape
                center_x, center_y = int(landmk.x * w), int(landmk.y * h)
                # print(id, center_x, center_y)

                landmark_list.append([id, center_x, center_y])
                if draw:
                    cv.circle(frame, (center_x, center_y), 1, color=(255,0,0), thickness=-1, lineType=cv.FONT_HERSHEY_PLAIN)

        return landmark_list

