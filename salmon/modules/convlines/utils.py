import numpy as np
import math
from skimage import measure
from skimage.morphology import thin, disk, closing, remove_small_objects

def find_conv_lines(u, v):
    """
    Identify convergence lines in a 2D wind field using Weller et al. (2017) method.
    Adapted from the original function by Caroline Bain and Prince Xavier.
    """
    # === Parameters ===
    conv_min = 0.5
    search_box = 11  # MUST be odd
    deltas = 1
    s_floor = search_box // 2
    s_ceil = -(-search_box // 2)  # Ceiling division

    if u.ndim != 2 or v.ndim != 2:
        return None, None, None

    # === Step 1: Convergence computation ===
    conv_orig = -1 * (np.gradient(u, axis=1) + np.gradient(v, axis=0))

    # === Step 2: Smoothing ===
    padded = np.pad(conv_orig, ((1, 1), (1, 1)), mode='constant')
    conv_sm = (
        4 * conv_orig +
        padded[2:, 1:-1] + padded[:-2, 1:-1] +
        padded[1:-1, 2:] + padded[1:-1, :-2]
    ) / 8

    # === Step 3: Thresholding ===
    binary_conv = (conv_sm > conv_min).astype(float)
    binary_conv[:s_floor, :] = 0
    binary_conv[:, :s_floor] = 0
    binary_conv[-s_floor:, :] = 0
    binary_conv[:, -s_floor:] = 0
    convergence_lines = np.zeros_like(binary_conv)

    # === Step 4: Object labeling ===
    labels = measure.label(binary_conv)
    num_labels = labels.max()

    x = np.arange(search_box)
    dist = np.abs(x - s_floor)
    Xdist, Ydist = np.meshgrid(dist, dist)

    for label in range(1, num_labels + 1):
        mask = labels == label
        this_blob = conv_sm * mask
        rows, cols = np.nonzero(mask)

        for r, c in zip(rows, cols):
            mm = this_blob[r - s_floor:r + s_ceil, c - s_floor:c + s_ceil]
            if mm.shape != (search_box, search_box):
                continue  # skip borders

            # Geometric analysis via point of inertia
            Rx = np.sum(Xdist * mm) / np.sum(mm) if np.sum(mm) > 0 else 0
            Ry = np.sum(Ydist * mm) / np.sum(mm) if np.sum(mm) > 0 else 0
            # Simplified version of the original eigenvalue computation
            a = np.sum(mm * (Xdist - Rx) ** 2)
            b = np.sum(mm * (Ydist - Ry) ** 2)
            c_term = np.sum(mm * (Xdist - Rx) * (Ydist - Ry))

            omega = 0.5 * math.sqrt((a - b) ** 2 + 4 * c_term ** 2)
            eigval1 = 0.5 * (a + b) + omega
            eigval2 = 0.5 * (a + b) - omega

            if eigval1 == 0: continue
            if abs(1 - abs(eigval2 / eigval1)) < 0.2:
                continue  # too circular

            f_mid = this_blob[r, c]
            
            # Profile analysis along minor axis
            if omega == abs(0.5 * (a - b)):
                if eigval1 <= eigval2:
                    f_pos, f_neg = conv_sm[r + 1, c], conv_sm[r - 1, c]
                else:
                    f_pos, f_neg = conv_sm[r, c + 1], conv_sm[r, c - 1]
            else:
                alpha = 0.5 * math.atan2(2 * c_term, a - b)
                dx, dy = math.sin(alpha) * deltas, math.cos(alpha) * deltas

                def bilinear_interp(rr, cc):
                    # Basic bilinear interp
                    i, j = int(rr), int(cc)
                    di, dj = rr - i, cc - j
                    return (
                        conv_sm[i, j] * (1 - di) * (1 - dj) +
                        conv_sm[i + 1, j] * di * (1 - dj) +
                        conv_sm[i, j + 1] * (1 - di) * dj +
                        conv_sm[i + 1, j + 1] * di * dj
                    )

                f_pos = bilinear_interp(r + dy, c + dx)
                f_neg = bilinear_interp(r - dy, c - dx)

            Ggrad = (f_pos - 2 * f_mid + f_neg) / (2 * deltas ** 2)
            Gtran = (f_pos - f_neg) / (2 * deltas)
            if Ggrad == 0: continue
            smax = -Gtran / (2 * Ggrad)

            if abs(smax) < 0.5 * math.sqrt(2):
                convergence_lines[r, c] = 1

    # === Final image cleanup ===
    lines_thin = thin(convergence_lines)
    joined = closing(lines_thin, disk(2)) + convergence_lines
    joined[joined > 1] = 1
    cleaned = remove_small_objects(thin(joined).astype(bool), 3).astype(int)

    return conv_orig, convergence_lines, cleaned
