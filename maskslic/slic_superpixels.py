# coding=utf-8
from __future__ import division, absolute_import, unicode_literals, print_function
import collections as coll
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

from maskslic.processing import get_mpd
import warnings
import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries


from skimage.util import img_as_float, regular_grid
from maskslic._slic import (_slic_cython,
                            _enforce_label_connectivity_cython)
from skimage.color import rgb2lab


def place_seed_points(image, img, mask, n_segments, spacing, q=99.99):

    """
    Method for placing seed points in an ROI

    Note:
    Optimal point placement problem is somewhat related to the k-center problem
     metric facility location (MFL)
     Maxmin facility location
    https://en.wikipedia.org/wiki/Facility_location_problem

    Parameters
    ----------
    image
    mask
    n_segments
    spacing

    Returns
    -------

    """

    segments_z = np.zeros(n_segments, dtype=np.int64)
    segments_y = np.zeros(n_segments, dtype=np.int64)
    segments_x = np.zeros(n_segments, dtype=np.int64)

    m_inv = np.copy(mask)

    # SEED STEP 1:  n seeds are placed as far as possible from every other seed and the edge.

    theta = 0

    for ii in range(n_segments):

        # distance transform

        dtrans = distance_transform_edt(m_inv, sampling=spacing)
        dtrans = gaussian_filter(dtrans, sigma=2.0)

        # dtransg = ndi.gaussian_gradient_magnitude(dtrans, sigma=2.0)
        # plt.figure()
        # plt.imshow(mask[0, :, :])
        # plt.figure()
        # plt.imshow(m_inv[0, :, :])
        # plt.show()

        perc1 = np.percentile(dtrans, q=q)
        mask_dtrans = dtrans > perc1
        pdtrans, nb_labels = ndi.label(mask_dtrans)

        # plt.figure()
        # plt.imshow(pdtrans[0, :, :])
        # plt.show()

        if ii < 2:
            sizes = ndi.sum(mask_dtrans, pdtrans, range(nb_labels + 1))
            # Use the maximum locations for the first two points
            coords1 = np.nonzero(pdtrans == np.argmax(sizes))
            segments_z[ii] = round(np.mean(coords1[0]))
            segments_x[ii] = round(np.mean(coords1[1]))
            segments_y[ii] = round(np.mean(coords1[2]))

        else:
            # Define a vector that is used to produce a reference frame
            u = np.array([segments_x[1] - segments_x[0], segments_y[1] - segments_y[0]])
            u = u / np.sqrt(np.sum(u**2))

            phi = np.zeros((nb_labels,))
            for vv in range(nb_labels):
                coords1 = np.nonzero(pdtrans == (vv+1))
                v = np.array([np.mean(coords1[1]) - segments_x[0], np.mean(coords1[2] - segments_y[0])])
                v = v / np.sqrt(np.sum(v**2))

                # Directed angle
                va = np.arctan2(v[1], v[0])
                if va < 0:
                    va += 2*np.pi
                ua = np.arctan2(u[1], u[0])
                if ua < 0:
                    ua += 2*np.pi

                phi[vv] = va - ua
                if phi[vv] < 0:
                    phi[vv] += 2*np.pi

            # Difference between previous theta and current theta
            phidiff = phi - theta
            # print("phidiff 1: ", phidiff)

            phidiff += (phidiff < 0) * 2*np.pi
            # print("phidiff 2: ", phidiff)
            iphi = np.argmin(phidiff)

            theta = phi[iphi]
            # print("theta: ", theta)

            coords1 = np.nonzero(pdtrans == (iphi+1))
            segments_z[ii] = round(np.mean(coords1[0]))
            segments_x[ii] = round(np.mean(coords1[1]))
            segments_y[ii] = round(np.mean(coords1[2]))

            # Calculate a reference vector for each candidate
            # Calculate each candidate angle
            # Choose the next greatest angle in the anti-clockwise direction


        # plt.figure()
        # plt.imshow(pdtrans[0, :, :])
        # plt.show()


        # mcoords = np.nonzero(dtrans == np.max(dtrans))
        #
        # # TODO: How about maximising the summed distance from all other points? Or n closest points as a
        # # TODO: way to get rid of the redundancy of having multiple choices?
        # # TODO: midpoint of the largest connected line
        #
        # if len(mcoords[0]) > 1:
        #     print("X:", mcoords[1])
        #     print("Y:", mcoords[2])
        #
        #     if ii == 0:
        #         print("shit")
        #         continue
        #
        #     d2 = np.zeros((len(mcoords[0]),))
        #     for cc in range(len(mcoords[0])):
        #         d2[cc] = np.sum(np.sqrt((segments_x - mcoords[1][cc])**2 + (segments_y - mcoords[2][cc])**2))
        #
        #     # Select value that maximises the distance
        #     p1 = np.argmax(d2)
        #
        #     # plt.figure()
        #     # plt.imshow(image[0, :, :, 1])
        #     # plt.plot(mcoords[2], mcoords[1], 'ro')
        #     # plt.plot(mcoords[2][p1], mcoords[1][p1], 'go')
        #     # plt.show()
        #
        #     segments_z[ii] = mcoords[0][p1]
        #     segments_x[ii] = mcoords[1][p1]
        #     segments_y[ii] = mcoords[2][p1]
        #
        # else:
        #
        #     segments_z[ii] = mcoords[0][0]
        #     segments_x[ii] = mcoords[1][0]
        #     segments_y[ii] = mcoords[2][0]

        # adding a new point
        m_inv[segments_z[ii], segments_x[ii], segments_y[ii]] = False

        # Plot: Illustrate the seed point selection method

        # plt.figure()
        # plt.imshow(img)
        # my_cmap = plt.cm.get_cmap('jet')  # get a copy of the gray color map
        # my_cmap.set_bad(alpha=0)  # s
        # d11 = dtrans[segments_z[ii], :, :]
        # d11[d11==0] = np.nan
        # plt.imshow(d11, cmap=my_cmap)
        # plt.contour(mask[segments_z[ii], :, :] == 1, contours=1, colors='red', linewidths=1)
        # plt.plot(segments_y[ii], segments_x[ii], marker='o', color='green')
        # plt.axis('off')
        # plt.show()

    segments_color = np.zeros((segments_z.shape[0], image.shape[3]))
    segments = np.concatenate([segments_z[..., np.newaxis],
                               segments_x[..., np.newaxis],
                               segments_y[..., np.newaxis],
                               segments_color], axis=1)

    sz = np.ascontiguousarray(segments_z, dtype=np.int32)
    sx = np.ascontiguousarray(segments_x, dtype=np.int32)
    sy = np.ascontiguousarray(segments_y, dtype=np.int32)

    out1 = get_mpd(sz, sx, sy)
    step_z, step_x, step_y = out1[0], out1[1], out1[2]

    return segments, step_x, step_y, step_z


def slic(image, n_segments=100, compactness=10., max_iter=10, sigma=0,
         spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=False, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False, seed_type='grid', mask=None, recompute_seeds=False, 
         plot_examples=False):
    """Segments image using k-means clustering in Color-(x,y,z) space.

    Parameters
    ----------
    image : 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.
    compactness : float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    max_iter : int, optional
        Maximum number of iterations of k-means.
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    convert2lab : bool, optional
        Whether the input should be converted to Lab colorspace prior to
        maskslic. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    enforce_connectivity: bool, optional
        Whether the generated segments are connected or not
    min_size_factor: float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    max_size_factor: float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    slic_zero: bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Supervoxel analysis will only be performed on points at
        which mask == True

    Returns
    -------
    labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      maskslic.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing.

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Süsstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO

    Examples
    --------
    >>> from maskslic import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)

    Increasing the compactness parameter yields more square regions:

    >>> segments = slic(img, n_segments=100, compactness=20)

    """

    if enforce_connectivity:
        raise NotImplementedError("Enforce connectivity has not been implemented yet for maskSLIC.\n"
                                  "Please set enforce connectivity to 'False' ")

    if slic_zero:
        raise NotImplementedError("Slic zero has not been implemented yet for maskSLIC.")


    img = np.copy(image)
    if mask is not None:
        msk = np.copy(mask==1)
    else:
        msk = None
    # print("mask shape", msk.shape)

    if mask is None and seed_type == 'nplace':
        warnings.warn('nrandom assignment of seed points should only be used with an ROI. Changing seed type.')
        seed_type = 'size'

    if seed_type == 'nplace' and recompute_seeds is False:
        warnings.warn('Seeds should be recomputed when seed points are randomly assigned')

    image = img_as_float(image)

    is_2d = False
    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if mask is None:
        mask = np.ones(image.shape[:3], dtype=np.bool)
    else:
        mask = np.asarray(mask, dtype=np.bool)

    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]

    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=np.double)
        sigma /= spacing.astype(np.double)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=np.double)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    depth, height, width = image.shape[:3]

    if seed_type == 'nplace':

        segments, step_x, step_y, step_z = place_seed_points(image, img, mask, n_segments, spacing)
        # print('{0}, {1}, {2}'.format(step_x, step_y, step_z))

    elif seed_type == 'grid':

        # initialize cluster centroids for desired number of segments
        # essentially just outputs the indices of a grid in the x, y and z direction
        grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
        # returns 3 slices (an object representing an array of slices, see builtin slice)
        slices = regular_grid(image.shape[:3], n_segments)
        step_z, step_y, step_x = [int(s.step) for s in slices]  # extract step size from slices
        segments_z = grid_z[slices]  # use slices to extract coordinates for centre points
        segments_y = grid_y[slices]
        segments_x = grid_x[slices]


        # mask_ind = mask[slices].reshape(-1)
        # list of all locations as well as zeros for the color features
        segments_color = np.zeros(segments_z.shape + (image.shape[3],))
        segments = np.concatenate([segments_z[..., np.newaxis],
                                   segments_y[..., np.newaxis],
                                   segments_x[..., np.newaxis],
                                   segments_color],
                                  axis=-1).reshape(-1, 3 + image.shape[3])

        if mask is not None:
            ind1 = mask[segments[:, 0].astype('int'), segments[:, 1].astype('int'), segments[:, 2].astype('int')]
            segments = segments[ind1, :]


        # seg_list = []
        # for ii in range(segments.shape[0]):
        #     if mask[segments[ii, 0], segments[ii, 1], segments[ii, 2]] != 0:
        #         seg_list.append(ii)
        # segments = segments[seg_list, :]
    else:
        raise ValueError('seed_type should be nrandom or grid')

    segments = np.ascontiguousarray(segments)

    # we do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    step = float(max((step_z, step_y, step_x)))
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image * ratio, dtype=np.double)
    mask = np.ascontiguousarray(mask, dtype=np.int32)

    segments_old = np.copy(segments)

    if recompute_seeds:

        # Seed step 2: Run SLIC to reinitialise seeds
        # Runs the supervoxel method but only uses distance to better initialise the method
        labels = _slic_cython(image, mask, segments, step, max_iter, spacing, slic_zero, only_dist=True)

    # Testing
    if plot_examples:
        fig = plt.figure()
        plt.imshow(img)
        if msk is not None:
            plt.contour(msk, contours=1, colors='yellow', linewidths=1)
        plt.scatter(segments_old[:, 2], segments_old[:, 1], color='green')
        plt.axis('off')

        fig = plt.figure()
        plt.imshow(img)
        if msk is not None:
            plt.contour(msk, contours=1, colors='yellow', linewidths=1)
        plt.scatter(segments[:, 2], segments[:, 1], color='green')
        plt.axis('off')

    # image = np.ascontiguousarray(image * ratio)

    labels = _slic_cython(image, mask, segments, step, max_iter, spacing, slic_zero, only_dist=False)

    if enforce_connectivity:
        segment_size = depth * height * width / n_segments
        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)
        labels = _enforce_label_connectivity_cython(labels,
													mask,
                                                    n_segments,
                                                    min_size,
                                                    max_size)

    if is_2d:
        labels = labels[0]

    return labels
