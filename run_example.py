from __future__ import division, absolute_import, unicode_literals, print_function

import matplotlib.pyplot as plt

import maskslic as seg
from skimage.segmentation import mark_boundaries
from skimage.io import imread
import time

img = imread('chelsea.png')
roi = imread('chelsea_mask.png')

# roi = img_as_float(chelsea_mask())
roi = roi[:, :, 3] > 0

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