import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
plt.rcParams['figure.figsize'] = [15, 10]

# videoPath = "data/CarFromDrone.mov"
videoPath = 'data/ManyCars.mov'
# videoPath = 0

current_region = None

x1 = 0
x2 = 0
y1 = 0
y2 = 0
width = 0
height = 0

state = 0
p1, p2 = (0, 0), (0, 0)
img, tracker, cap , template = None, {}, None, None
isTrackerInitialized = False


# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2, tracker, isTrackerInitialized, img, cap, x1, x2, y1, y2, template, width, height
    # Left click
    # Left button down
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        state = 1
    # Mouse move with left button down
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if state == 1:
            # Draw temporary rectangle while dragging
            temp_img = img.copy()
            cv2.rectangle(temp_img, p1, (x, y), (255, 0, 0), 2)
            cv2.imshow('Frame', temp_img)
    # Left button up
    elif event == cv2.EVENT_LBUTTONUP:
        p2 = (x, y)
        state = 2
        # Draw final rectangle
        final_img = img.copy()
        cv2.rectangle(final_img, p1, p2, (255, 0, 0), 2)
        cv2.imshow('Frame', final_img)
        # Initialize tracker with bounding box
        x1 = min(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        x2 = max(p1[0], p2[0])
        y2 = max(p1[1], p2[1])
        width = x2 - x1
        height = y2 - y1
        template = img[y1:y2, x1:x2] / 255
        isTrackerInitialized = True
    # Right button down
    elif event == cv2.EVENT_RBUTTONDOWN:
        state = 0
        # Clear all rectangles
        success, img = cap.read()
        img = imutils.resize(img, width=840)
        cv2.imshow('Frame', img)
        isTrackerInitialized = False


# width = x2 - x1
# height = y2 - y1


# Limit the search to a certain vicinity (since the cars can only move that fast)
search = 50

cap = cv2.VideoCapture(videoPath)
cap.set(3, 840)
cap.set(4, 680)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Register the mouse callback
cv2.setMouseCallback('Frame', on_mouse)
n_farmes = 0

while cap.isOpened():
    success, img = cap.read()
    # results = model(img, stream=True)
    if not success:
        break
    n_farmes = n_farmes + 1
    if n_farmes % 10 == 0:
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isTrackerInitialized:
            search_window = img[y1 - search:y2 + search, x1 - search:x2 + search] / 255
            # Tracking by minimising simple SAD (sum of absolute differences) loss
            # Equivalent to MSE loss (in **this** case) but faster
            track_x1 = None
            track_y1 = None
            loss = 1e6
            for r in range(0, search_window.shape[0] - height):
                for c in range(0, search_window.shape[1] - width):
                    candidate = search_window[r:r + height, c:c + width]
                    score = np.sum(np.abs(template - candidate))
                    if score < loss:
                        loss = score
                        track_x1 = c
                        track_y1 = r

                        # Update the bounding box of the tracked object
            x1 = x1 - search + track_x1
            y1 = y1 - search + track_y1
            print(x1, y1, width, height)

            # Show the tracker working
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 3)
            # plt.imshow(img)
            # plt.show(), plt.draw()
            # plt.waitforbuttonpress()
            # plt.clf()
            # img = imutils.resize(img, width=840)
            # if successTrack and state > 1:
            #     (x, y, w, h) = [int(a) for a in box]
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Frame', img)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
