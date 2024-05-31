import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]


# Open video file
cap = cv2.VideoCapture('data/lesson3.mp4')
assert cap.isOpened()

# Extract frames
frames = []
n_frames = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        n_frames = n_frames + 1

        # Keep every 20th frame
        if n_frames%20==0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.imshow(frames[-1])
            # plt.draw(), plt.show()
            # plt.waitforbuttonpress(0.1)
            # plt.clf()
    else:
        break


# When the everething done, release the videocapture object
cap.release()

#Apply Median filter across the time dimension
frames = np.array(frames)
result = np.median(frames, axis=0)

# Show result
plt.subplot(131), plt.imshow(frames[10, ...])
plt.subplot(132), plt.imshow(frames[50, ...])
plt.subplot(133), plt.imshow(result.astype(np.uint8))

plt.figure()
plt.subplot(121), plt.imshow(frames[30, ...])
plt.subplot(122), plt.imshow(result.astype(np.uint8))

plt.figure(), plt.imshow(result.astype(np.uint8))
plt.show()
