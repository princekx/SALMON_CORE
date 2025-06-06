
#!/usr/bin/env python
import numpy as np
import math
import functools
from skimage import measure
from skimage.morphology import thin, disk, closing, remove_small_objects
import scipy.ndimage as ndimage

def findConvLines_old(u, v):
    """
    Identify convergence lines in a 2D wind field using the method of Weller et al. (2017).

    This function computes convergence lines from gridded 2D wind fields (u, v components)
    using a multi-step geometric and statistical approach:

    Steps:
    1. Compute a convergence field from the input wind components.
    2. Apply a neighborhood smoothing filter to the convergence field.
    3. Threshold the convergence field to retain only positive values (converging flow).
    4. Label and isolate connected regions ("objects") of convergence.
    5. For each object:
       a. Analyze each pixel using an 11×11 surrounding window.
       b. Compute the point of inertia and derive the principal axes via eigenvectors.
       c. Evaluate the profile along the minor eigenvector axis.
       d. Fit a quadratic (binomial) curve to the three points along this axis.
       e. Determine if the peak of the binomial lies near the center (±0.5).
       f. If so, classify the point as part of a convergence line.
    6. Compile all such points into a convergence line map.
    7. Clean up the result by removing very small objects (less than 2 pixels in size).

    Parameters
    ----------
    u : 2D array (float)
        Zonal wind component (m/s). Must be gridded and match the shape of `v`.
    v : 2D array (float)
        Meridional wind component (m/s). Must be gridded and match the shape of `u`.

    Returns
    -------
    ConvOrig : 2D array
        The original convergence field calculated from `u` and `v`.
    OrigConvLines : 2D binary array
        A mask (1s and 0s) identifying raw convergence line pixels.
    CleanConvLines : 2D binary array
        The cleaned and connected convergence lines after removing small objects.

    Notes
    -----
    - Input data must be 2D (i.e., instantaneous horizontal wind field).
    - Recommended input: 10m wind data, though any vertical level can be used.
    - Function assumes input wind data are on the same horizontal grid.
    - Compatible with Python ≥3.6.

    References
    ----------
    Weller, H., et al. (2017). "Identification of convergence lines in wind fields."
    [Specific citation details needed]

    Examples
    --------
    >>> ConvOrig, OrigConvLines, CleanConvLines = findConvLines(u, v)

    Author
    ------
    Caroline Bain (caroline.bain@metoffice.gov.uk), 15 Jan 2019

    Modified
    ---------
    Prince Xavier (prince.xavier@metoffice.gov.uk), 21 May 2025
    """

    ### HARDWIRED Variables Change here! #####
    convmin = 0.5   # Minimum convergence will look at, based on GA6
    searchbox = 11  # Size of search grid ****!!!! MUST BE ODD NUMBER !!!!!*****
    deltas = 1  # looking this much either side of central point
    # note I found that search box doesn't actually make a huge difference to end result
    ### The following variables are referred to later and don't change:
    s_floor = np.int(np.floor(searchbox/2))
    s_ceil = np.int(np.ceil(searchbox/2))
    Xdist = np.abs(np.array(range(searchbox))-s_floor)
    Ydist = np.vstack(Xdist)
    XY = (Xdist*np.ones((searchbox,searchbox)))*(np.ones((searchbox,searchbox))*Ydist)

    #### (1) Make Convergence ########################################
    if u.ndim!=2 or v.ndim!=2:
        print('Your u or v has the wrong dimensions, should have u.ndim = 2')
        print('Returning you to your program without convergence or lines - try again!')
        return
    print('making convergence new method')
    # Note the below assumes cartesian equi-distant grid which works fine in small tropical domains.
    # Further code can be substituted here for spherical coodinates or irregular grid spacing.
    dudx = np.gradient(u, axis=1)
    dvdy = np.gradient(v, axis=0)
    divg = dudx + dvdy
    convorig = -1. * divg

    #### (2) Smoothing - I'm using Michaels cross-method here, there are several options here
    ####     my long term favourite would be to use img2 = skimage.restoration.denoise_tv_chambolle
    #img_gaus = ndimage.filters.gaussian_filter(convorig, 1, mode='nearest')
    #img2 = skimage.restoration.denoise_tv_chambolle(convorig, weight=0.4)
    conv_up = np.concatenate((convorig[1:,:],np.zeros((1,u.shape[1]))),axis=0)
    conv_dn = np.concatenate((convorig[:-1,:],np.zeros((1,u.shape[1]))),axis=0)
    conv_lf = np.concatenate((convorig[:,1:],np.zeros((u.shape[0],1))),axis=1)
    conv_rt = np.concatenate((convorig[:,:-1],np.zeros((u.shape[0],1))),axis=1)
    conv_sm = (4*convorig + conv_up + conv_dn + conv_lf + conv_rt)/8

    #### (3) Threshold convergence so only have +ve values ###########
    bin_c = conv_sm*1.
    bin_c = (bin_c<convmin).choose(bin_c,0)
    convergence = bin_c*1.            # take a copy of inside blobs to get convergence inside
    bin_c = (bin_c>0).choose(bin_c,1) # binary version of convergence
    # Remove edges so that the moving matrix works
    bin_c[:s_floor,:] = 0.
    bin_c[:,:s_floor] = 0.
    bin_c[:,-s_floor:] = 0.
    bin_c[-s_floor:,:] = 0.
    convergence_lines = np.zeros(convergence.shape) # set OUTPUT MATRIX

    #### (4) Identify individual 'objects' of convergence
    L = measure.label(bin_c)
    NUM = np.amax(L)
    for objects in list(range(NUM)):
        thisblob = convergence*( L==(objects+1) )
        [T,N] = (L==(objects+1)).nonzero()
        #(5) For each object, interrogate each pixel and look at an 11x11 box around it
        #    for each gridpoint [T[gridpt],N[gridpt]]
        for gridpt in list(range(len(T))):
            moving_matrix = thisblob[T[gridpt]-s_floor:T[gridpt]+s_ceil,N[gridpt]-s_floor:N[gridpt]+s_ceil]
            ## (6) Find intertia tensor (replacing mass with convergence) e.g. using sum of mr^2
            ## i.e. find [a  b; b  c], where:
            ## a = SUM(c_i.x_i^2)
            ## b = SUM(c_i.y_i^2)
            ## c = SUM(c_i.x_i.y_i)
            # (a, b and c should give a single number value each)
            Rx = np.sum(Xdist*moving_matrix)
            Ry = np.sum(Ydist*moving_matrix)
            a = np.sum(moving_matrix*((Xdist-Rx)**2))
            b = np.sum(moving_matrix*((Ydist-Ry)**2))
            XY=((Xdist-Rx)*np.ones((searchbox,searchbox)))*(np.ones((searchbox,searchbox))*(Ydist-Ry))
            c = np.sum(moving_matrix*XY)
            omega = 0.5*math.sqrt((a-c)**2+4*b**2)
            # (7) Getting eigenvalues to find what the minor axis is (the smallest value)
            eigenval1 = 0.5*(a+c) + omega
            eigenval2 = 0.5*(a+c) - omega
            eigenvector1 = np.vstack([ b/( 0.5*(a-c) - omega),-1])
            eigenvector2 = np.vstack([ b/( 0.5*(a-c) + omega),-1])
            # If loop to get rid of too circular points:
            if np.abs(1-np.abs(eigenval2/eigenval1))<0.2:
                #print('this point is too circular and blobby, gonna skip it')
                continue
            # If solution lies along axis don't work out eigenvectors(divide by 0)....
            fmid = thisblob[T[gridpt], N[gridpt]]
            if omega == np.abs(0.5*(a-c)) and eigenval1<=eigenval2: # one solution lies along y axis #
                #print('value is on the y axis')
                fpos = thisblob[T[gridpt]+1,N[gridpt]]
                fneg = thisblob[T[gridpt]-1,N[gridpt]]
            if omega == np.abs(0.5*(a-c)) and eigenval1>eigenval2: # one solution lies along x axis
                #print('value is on the x axis')
                fpos = thisblob[T[gridpt],N[gridpt]+1]
                fneg = thisblob[T[gridpt],N[gridpt]-1]
            # Find angle alpha between y and minor vector for all other solutions :
            if omega != np.abs(0.5*(a-c)):
                alpha = 0.5*math.atan((2*b)/(a-c)) #in radians
                # (8) Look along the minor eigenvector axis +/-1
                # Do bilinear interpolation to find values of fpos and fneg
                dx = math.sin(alpha)*deltas  # CHECK!! swap cos sin over???
                dy = math.cos(alpha)*deltas
                ### due to indexing if looking for more positive value must plus 1
                if alpha > 0: # alternatively eigenval1 > eigenval2:
                    f1 = convergence[T[gridpt]        , N[gridpt]] # because [rows,cols] or [x,y]
                    f2 = convergence[T[gridpt]+deltas, N[gridpt]]
                    f3 = convergence[T[gridpt]+deltas, N[gridpt]+deltas]
                    f4 = convergence[T[gridpt]        , N[gridpt]+deltas]
                    fpos = f1*(deltas-dy)*(deltas-dx) + f2*dy*(deltas-dx) + f3*dy*dx + f4*dx*(deltas-dy)
                    f2 = convergence[T[gridpt]-deltas , N[gridpt]]
                    f3 = convergence[T[gridpt]-deltas , N[gridpt]-deltas]
                    f4 = convergence[T[gridpt]        , N[gridpt]-deltas]
                    fneg = f1*(deltas-dy)*(deltas-dx) + f2*dy*(deltas-dx) + f3*dy*dx + f4*dx*(deltas-dy)
                if alpha < 0: # eigenval1 < eigenval2: # alternatively
                    f1 = convergence[T[gridpt]        , N[gridpt]] # because [rows,cols] or [x,y]
                    f2 = convergence[T[gridpt]+deltas,N[gridpt]]
                    f3 = convergence[T[gridpt]+deltas,N[gridpt]-deltas]
                    f4 = convergence[T[gridpt]        , N[gridpt]-deltas]
                    fpos = f1*(deltas-dy)*(deltas-dx) + f2*dy*(deltas-dx) + f3*dy*dx + f4*dx*(deltas-dy)
                    f2 = convergence[T[gridpt]-deltas , N[gridpt]]
                    f3 = convergence[T[gridpt]-deltas , N[gridpt]+deltas]
                    f4 = convergence[T[gridpt]        , N[gridpt]+deltas]
                    fneg = f1*(deltas-dy)*(deltas-dx) + f2*dy*(deltas-dx) + f3*dy*dx + f4*dx*(deltas-dy)
            # (9) Fit a binomial to those three points
            # Now you find the polynomial curve across fneg, fmid and fpos. The max point of the
            # curve will be where the convergence line is. If it isn't in this domain, keep looking
            Ggrad = (1/(2*deltas**2))*(fpos - 2*fmid + fneg)
            Gtran = (1/(2*deltas))*(fpos - fneg)
            smax = (-1*Gtran) / (2*Ggrad)
            # (10) Look for the maximum of the binomial: if it is +/-0.5 from the central point it is
            #      classed as 'on' and point is identified as convergence line
            if np.abs(smax)<(0.5*math.sqrt(2)): # this is the max point
                convergence_lines[T[gridpt],N[gridpt]] = 1
                #xval = smax*math.cos(alpha)+N # exact x location
                #yval = smax*math.sin(alpha)+T # exact y location
                #print('found curve maximum point! Its at:',T[gridpt],N[gridpt])
            # if the curve max point is outside, disregard this point


    ####### Line joining/ sorting method (to replicate Gareth's sort_lines): ##########
    # The next 4 steps are light image processing and could be removed but does help clean up noisy images
    # This will join lines, reduce 'double lines' close together and get rid of small bits:
    # (1) Thin lines so only 1 pixel wide - finds the mid point of a plateau
    thinned_lines = thin(convergence_lines)
    # (2) Using closing (dilation followed by erosion) to join nearby points
    selem = disk(2) # looking in a neighborhood of 2
    cloLines = closing(thinned_lines, selem)
    comLines = cloLines+convergence_lines
    comLines[comLines>1]=1
    # (3) Doing one last check to ensure there are no double points
    thincomLines = thin(comLines)
    ar = measure.label(thincomLines)
    # (4) Get rid of small bits in the image < 3 pixels
    cleanLines = remove_small_objects(ar,3)
    cleanLines[cleanLines>1]=1

    ############## Output ####################
    # Returning the convergence field and the lines
    return convorig, convergence_lines, cleanLines

def findConvLines(u, v):
    """
    Modified
    --------
    ChatGPT 21 May 2025

    Identify convergence lines in a 2D wind field using Weller et al. (2017) method.
    See detailed docstring in the original version for methodology.

    This function computes convergence lines from gridded 2D wind fields (u, v components)
    using a multi-step geometric and statistical approach:

    Steps:
    1. Compute a convergence field from the input wind components.
    2. Apply a neighborhood smoothing filter to the convergence field.
    3. Threshold the convergence field to retain only positive values (converging flow).
    4. Label and isolate connected regions ("objects") of convergence.
    5. For each object:
       a. Analyze each pixel using an 11×11 surrounding window.
       b. Compute the point of inertia and derive the principal axes via eigenvectors.
       c. Evaluate the profile along the minor eigenvector axis.
       d. Fit a quadratic (binomial) curve to the three points along this axis.
       e. Determine if the peak of the binomial lies near the center (±0.5).
       f. If so, classify the point as part of a convergence line.
    6. Compile all such points into a convergence line map.
    7. Clean up the result by removing very small objects (less than 2 pixels in size).

    Parameters
    ----------
    u : 2D array (float)
        Zonal wind component (m/s). Must be gridded and match the shape of `v`.
    v : 2D array (float)
        Meridional wind component (m/s). Must be gridded and match the shape of `u`.

    Returns
    -------
    ConvOrig : 2D array
        The original convergence field calculated from `u` and `v`.
    OrigConvLines : 2D binary array
        A mask (1s and 0s) identifying raw convergence line pixels.
    CleanConvLines : 2D binary array
        The cleaned and connected convergence lines after removing small objects.

    Notes
    -----
    - Input data must be 2D (i.e., instantaneous horizontal wind field).
    - Recommended input: 10m wind data, though any vertical level can be used.
    - Function assumes input wind data are on the same horizontal grid.
    - Compatible with Python ≥3.6.

    References
    ----------
    Weller, H., et al. (2017). "Identification of convergence lines in wind fields."
    [Specific citation details needed]

    Examples
    --------
    >>> ConvOrig, OrigConvLines, CleanConvLines = findConvLines(u, v)

    Author
    ------
    Caroline Bain (caroline.bain@metoffice.gov.uk), 15 Jan 2019

    Modified
    ---------
    Prince Xavier (prince.xavier@metoffice.gov.uk), 29 May 2025
    """

    # === Parameters ===
    conv_min = 0.5
    search_box = 11  # MUST be odd
    deltas = 1
    s_floor = search_box // 2
    s_ceil = -(-search_box // 2)  # Ceiling division

    if u.ndim != 2 or v.ndim != 2:
        print("Input arrays must be 2D")
        return

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

            Rx = np.sum(Xdist * mm)
            Ry = np.sum(Ydist * mm)
            a = np.sum(mm * (Xdist - Rx) ** 2)
            b = np.sum(mm * (Ydist - Ry) ** 2)
            c_term = np.sum(mm * (Xdist - Rx) * (Ydist - Ry))

            omega = 0.5 * math.sqrt((a - b) ** 2 + 4 * c_term ** 2)
            eigval1 = 0.5 * (a + b) + omega
            eigval2 = 0.5 * (a + b) - omega

            if abs(1 - abs(eigval2 / eigval1)) < 0.2:
                continue  # too circular

            f_mid = this_blob[r, c]
            dx = dy = 0
            f_pos = f_neg = 0

            if omega == abs(0.5 * (a - b)):
                # aligned with axis
                if eigval1 <= eigval2:
                    f_pos = conv_sm[r + 1, c]
                    f_neg = conv_sm[r - 1, c]
                else:
                    f_pos = conv_sm[r, c + 1]
                    f_neg = conv_sm[r, c - 1]
            else:
                alpha = 0.5 * math.atan2(2 * c_term, a - b)
                dx, dy = math.sin(alpha) * deltas, math.cos(alpha) * deltas

                # Bilinear interpolation here (you could wrap this in a helper function)
                def bilinear_interp(rr, cc):
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
            smax = -Gtran / (2 * Ggrad)

            if abs(smax) < 0.5 * math.sqrt(2):
                convergence_lines[r, c] = 1

    # === Final image cleanup (Steps 7+) ===
    lines_thin = thin(convergence_lines)
    joined = closing(lines_thin, disk(2)) + convergence_lines
    joined[joined > 1] = 1
    cleaned = remove_small_objects(thin(joined).astype(bool), 3).astype(int)

    return conv_orig, convergence_lines, cleaned
