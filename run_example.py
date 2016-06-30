
import matplotlib.pyplot as plt

import maskslic as seg
from skimage.segmentation import mark_boundaries
from skimage.io import imread

img = imread('chelsea.png')
roi = imread('chelsea_mask.png')

# roi = img_as_float(chelsea_mask())
roi = roi[:, :, 3] > 0

segments = seg.slic(img, compactness=30, seed_type='nplace', mask=roi, n_segments=12,
                    recompute_seeds=True, plot_examples=True)

plt.figure()
plt.title('maskSLIC')
plt.imshow(mark_boundaries(img, segments))
plt.contour(roi, contours=1, colors='red', linewidths=1)
plt.axis('off')


segments = seg.slic(img, compactness=30, seed_type='grid', n_segments=80, plot_examples=False)
# segments[roi==0] = -1

plt.figure()
plt.title('Conventional SLIC')
plt.imshow(mark_boundaries(img, segments))
plt.contour(roi, contours=1, colors='red', linewidths=1)
plt.axis('off')
plt.show()

