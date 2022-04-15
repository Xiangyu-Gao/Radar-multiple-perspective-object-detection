import numpy as np
import math
import argparse

from config import radar_configs


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def cart2pol(x, y):
    rho = (x * x + y * y) ** 0.5
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart_ramap(rho, phi):
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return x, y


def cart2pol_ramap(x, y):
    rho = (x * x + y * y) ** 0.5
    phi = np.arctan2(x, y)
    return rho, phi


def find_nearest(array, value):
    """
    Find nearest value to 'value' in 'array'
    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def chirp_amp(chirp):
    c0, c1, c2 = chirp.shape
    if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'RISEP':
        if c0 == 2:
            chirp_abs = np.sqrt(chirp[0, :, :] ** 2 + chirp[1, :, :] ** 2)
        elif c2 == 2:
            chirp_abs = np.sqrt(chirp[:, :, 0] ** 2 + chirp[:, :, 1] ** 2)
        else:
            raise ValueError
    elif radar_configs['data_type'] == 'AP' or radar_configs['data_type'] == 'APSEP':
        if c0 == 2:
            chirp_abs = chirp[0, :, :]
        elif c2 == 2:
            chirp_abs = chirp[:, :, 0]
        else:
            raise ValueError
    else:
        raise ValueError
    return chirp_abs


def prob2image(prob_array):
    prob_array[np.where(prob_array < 0)] = 0
    prob_array[np.where(prob_array > 1)] = 1
    return np.array(prob_array * 255, dtype=int)


def detect_peaks(image, threshold=0.3):

    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            # print(h, w)
            area = image[h-1:h+2, w-2:w+3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] <= 2 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def dist_point_segment(point, segment):
    x3, y3 = point
    (x1, y1), (x2, y2) = segment

    px = x2-x1
    py = y2-y1
    norm = px*px + py*py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5

    return dist, (x, y)


def rotate_conf_pattern(dx, dy, ori):
    dr = (dx * dx + dy * dy) ** 0.5
    dtheta = math.atan2(dy, dx)
    dtheta -= ori
    dx_new = dr * math.cos(dtheta)
    dy_new = dr * math.sin(dtheta)
    return dx_new, dy_new


# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def is_inside_triangle(p1, p2, p3, p):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p

    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)
    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)
    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)
    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if A == A1 + A2 + A3:
        return True
    else:
        return False
