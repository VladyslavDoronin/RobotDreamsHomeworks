import cv2
from time import perf_counter


# videoPath = "data/CarFromDrone.mov"
# videoPath = 'data/ManyCars.mov'
videoPath = '/home/user/Downloads/Telegram Desktop/1.mp4'
# videoPath = 0

current_region = None
x1, x2, y1, y2 = 0, 0, 0, 0
width, height = 0, 0
boxes = []

state = 0
n_frames = 0
search = 50
isPaused = False
isTrackerInitialized = False
p1, p2 = (0, 0), (0, 0)
img, tracker, cap, bbox, ok = None, None, None, None, None

# Set up tracker
tracker_types = ['MIL', 'KCF', 'CSRT']
# tracker_type = tracker_types[0]
# tracker_type = tracker_types[1]
tracker_type = tracker_types[2]

# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    global state, p1, p2, tracker, isTrackerInitialized, img, cap, x1, x2, y1, y2, width, height, bbox, ok
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
        if isTrackerInitialized:
            if tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()

            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()

            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            # Clear all rectangles
            success, img = cap.read()
            isTrackerInitialized = False
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

        # Initialize tracker
        bbox = (x1, y1, width, height)
        ok = tracker.init(img, bbox)
        isTrackerInitialized = True
    # Right button down
    elif event == cv2.EVENT_RBUTTONDOWN:
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()

        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()

        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        state = 0
        # Clear all rectangles
        success, img = cap.read()
        cv2.imshow('Frame', img)
        isTrackerInitialized = False

if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()

if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()

if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

cap = cv2.VideoCapture(videoPath)
cap.set(3, 840)
cap.set(4, 680)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Register the mouse callback
cv2.setMouseCallback('Frame', on_mouse)

while cap.isOpened():
    if not isPaused:
        success, img = cap.read()
        # results = model(img, stream=True)
        if not success:
            break

        n_frames = n_frames + 1
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isTrackerInitialized:
            start = perf_counter()
            ok, bbox = tracker.update(img)
            # print(ok, bbox)

            # Show the tracker working
            x1, y1 = bbox[0], bbox[1]
            width, height = bbox[2], bbox[3]

            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            elapsed = perf_counter() - start
            print('Elapsed time:', elapsed)

        cv2.imshow('Frame', img)

    key = cv2.waitKey(24)
    if key == ord('q'):
        break
    elif key == ord('p'):
        isPaused = not isPaused

cap.release()
cv2.destroyAllWindows()


# Compare the results:
# Do you see any differences? If so, what are they?
# Does one tracker perform better than the other? In what way?

# И так, тут я включал как просто скачанные видео, так и камеру. Выбирал разные объекты с разным уровнем освещения

# Разница во всех трех методах довольно большая. Особенно большая разница оп времени.
# Для метода KCF, на мое пк, это занимает в среднем 0.005-0.01 секунду. Он самый быстрый
# Для CSRT - 0.018-0.025. Из больших плюсов - адаптируется под изменения масштаба и размеров самого объекта.
# MIL оказался самый медленный и по сути сказал бы бесполезный. Для моих видео он мог терять объекты, и не масштабировался. По времени заняло 0.06-0.08

# Еще заметил, что довольно сильно влияет сам размер обрабатываемой картинки. Поэтому выставил
# cap.set(3, 840)
# cap.set(4, 680)
# До этого не применял ничего по размера и работал даже CSRT хуже - раньше мог быстрее терять объекты, особенно чем быстрее они перемещались.
# Потом попробовал выставить 1280х1024 и результат стал еще хуже

