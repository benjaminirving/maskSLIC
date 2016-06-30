"""

Cython wrapper for some processing scripts

(c) Benjamin Irving 20150713

"""

from __future__ import division, print_function, absolute_import

from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp

cdef extern from "processing.h":
    vector[int] get_mean_point_distance(vector[int] &, vector[int] &, vector[int] &)

def get_mpd(cnp.ndarray[int, ndim=1] x, cnp.ndarray[int, ndim=1] y, cnp.ndarray[int, ndim=1] z):
    """

    Get the maximum distance between two points in the region

    :return:
    """

    dist = get_mean_point_distance(x, y, z)
    return np.array(dist)