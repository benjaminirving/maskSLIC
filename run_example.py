from __future__ import division, absolute_import, unicode_literals, print_function

import matplotlib.pyplot as plt

import maskslic as seg
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.io import imread
import time

img = imread('chelsea.png')

# The ROI is also stored as an image for viewing convenience
# But the roi input input maskSLIC should be a binary image with the same spatial
# Dimensions as the image (in this case 300x451)
# roi = imread('chelsea_mask.png')
# The alpha channel is used to store the ROI in this case and is converted into a logical array of 0s and 1s
# roi = roi[:, :, 3] > 0

# Alternatively a mask could be created manually with for example:
roi = np.zeros((img.shape[0], img.shape[1]))
a, b = 150, 150
r = 100
y,x = np.ogrid[-a:img.shape[0]-a, -b:img.shape[1]-b]
mask = x*x + y*y <= r*r
roi[mask] = 1


# ~~~~~~~~~~~~ Example 1: maskSLIC ~~~~~~~~~~~~~

t1 = time.time()
# Note that compactness is defined differently because a grid is not used. Lower compactness for maskSLIC is equivalent
segments = seg.slic(img, compactness=10, seed_type='nplace', mask=roi, n_segments=12,
                    recompute_seeds=True, plot_examples=True, enforce_connectivity=True)
print("Time: {:.2f} s".format(time.time() - t1))

plt.figure()
plt.title('maskSLIC')
plt.imshow(mark_boundaries(img, segments))
plt.contour(roi, contours=1, colors='red', linewidths=1)
plt.axis('off')

# ~~~~~~~~~~~ Example 2: SLIC ~~~~~~~~~~~~~~~~~

t1 = time.time()
segments = seg.slic(img, compactness=30, seed_type='grid', n_segments=80, plot_examples=False, enforce_connectivity=True)
# segments[roi==0] = -1
print("Time: {:.2f} s".format(time.time() - t1))

plt.figure()
plt.title('Conventional SLIC')
plt.imshow(mark_boundaries(img, segments))
plt.contour(roi, contours=1, colors='red', linewidths=1)
plt.axis('off')
plt.show()

plt.show()