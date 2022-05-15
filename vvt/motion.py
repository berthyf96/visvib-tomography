import datetime
import os
import threading
import time

import cv2
import numpy as np
import pyrtools as pt
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

import vvt.utils as utils

def interpolate_2d_motion(points, motion_2d, res=100, is_fill_nan=True):
    '''Interpolate a 2D motion field between known motion at given points.
    Inputs:
        points -- array of size (N_PTS, 2) with the coordinates of
            the known points.
        motion_2d -- array of size (N_PTS, 2) with the (x, y) displacement
            for each of the known points.
        res -- number of pixels (in both x and y direction)
            in interpolation grid.
        is_fill_nan -- whether to 0-fill NaNs.
    Output:
        interp_dx -- interpolated horizontal displacement field.
        interp_dy -- interpolated vertical displacement field.
    '''
    if len(points) != len(motion_2d):
        raise ValueError('len(points) != len(motion_2d)')
    
    xmin = points[:,0].min()
    xmax = points[:,0].max()
    ymin = points[:,1].min()
    ymax = points[:,1].max()
    
    # Create image grid for interpolation.
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
    interp_dx = griddata(
        points, motion_2d[:,0], (grid_x, grid_y))
    interp_dy = griddata(
        points, motion_2d[:,1], (grid_x, grid_y))

    # Zero-fill NaNs.
    if is_fill_nan:
        interp_dx[np.isnan(interp_dx)] = 0
        interp_dy[np.isnan(interp_dy)] = 0

    return interp_dx, interp_dy

def project_and_interpolate_motion(points, motion_3d, proj_mat):
    '''Project 3d motion onto 2d space and interpolate motion fields.
    Inputs:
        points: Coordinates to plot on image space, of shape (N_PTS , 2).
        motion_3d: 3d motion field, of shape (N_PTS, 3).
        proj_mat: 3d-to-2d projection matrix, of shape (2, 3).
    Output:
        interp_dx: Interpolated horizontal motion field in image space.
        interp_dy: Interpolated vertical motion field in image space.
    '''
    if len(points) != len(motion_3d):
        raise ValueError('len(points) != len(motion_3d)')
    if proj_mat.shape[0] != 2 or proj_mat.shape[1] != 3:
        raise ValueError('proj_mat must be of shape (2, 3).')

    # Project 3d motion onto 2d image space.
    motion_2d = utils.project_points(motion_3d, proj_mat)

    # Interpolate 2d motion.
    interp_dx, interp_dy = interpolate_2d_motion(points, motion_2d)
    return interp_dx, interp_dy

def read_frames(vid_fn):
    '''Returns frames in specified GIF.'''
    if not os.path.exists(vid_fn):
        raise FileNotFoundError(vid_fn)
    
    reader = cv2.VideoCapture(vid_fn)
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in tqdm(range(n_frames), desc='Reading frames'):
        _, im = reader.read()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        frames.append(im)
    return frames

def delta_phase(phase, phase_ref):
    return np.mod(np.pi + (phase - phase_ref), 2 * np.pi) - np.pi

def im_dx(im):
    filt = np.array([0, -1, 1]).reshape(1, 3)
    return -fftconvolve(im, filt, mode='same')

def im_dy(im):
    filt = np.array([0, -1, 1]).reshape(3, 1)
    return -fftconvolve(im, filt, mode='same')

def amp_mask(amp, top_k=30):
    thresh = 0.5 * np.median(np.sort(amp.flatten())[::-1][:top_k])
    mask = (amp >= thresh).astype(float)
    return mask

def nonoutlier_mask(val_arr, pct=98):
    thresh = np.percentile(abs(val_arr), pct)
    mask = (abs(val_arr) <= thresh).astype(float)
    return mask

def _extract_motion_frame(frame, reference_frame, filter_pct, n_orients=2):
    '''Returns the horizontal and vertical displacement of given frame from
    reference frame.
    '''
    pyr_ref = pt.pyramids.SteerablePyramidFreq(
        reference_frame, order=n_orients-1, is_complex=True)
    pyr = pt.pyramids.SteerablePyramidFreq(
        frame, order=n_orients-1, is_complex=True)

    # Compute the change in phase.
    phase_h = np.angle(pyr.pyr_coeffs[(0,0)])
    dphase_h = delta_phase(phase_h, np.angle(pyr_ref.pyr_coeffs[(0,0)]))

    phase_v = np.angle(pyr.pyr_coeffs[(0,1)])
    dphase_v = delta_phase(phase_v, np.angle(pyr_ref.pyr_coeffs[(0,1)]))

    # Commpute displacement field.
    u = -(1/(im_dx(phase_h))) * dphase_h
    v = (1/(im_dy(phase_v))) * dphase_v

    # Filter outliers.
    mask_u = nonoutlier_mask(u, filter_pct)
    filtered_u = mask_u * u

    mask_v = nonoutlier_mask(v, filter_pct)
    filtered_v = mask_v * v

    return filtered_u, filtered_v

def _extract_motion_slice(motion_fields, slice_frames, slice_idxs,
                          reference_frame, filter_pct, thread_i):
    verbose = thread_i == 0
    slice_size = slice_idxs.stop - slice_idxs.start

    tic = time.time()
    for (i, frame) in enumerate(slice_frames):
        u, v = _extract_motion_frame(frame, reference_frame, filter_pct)
        motion_fields[0, slice_idxs.start + i, :, :] = u
        motion_fields[1, slice_idxs.start + i, :, :] = v

        if verbose and (i + 1) % 100 == 0:
            toc = time.time() - tic
            msg = '[Thread %d] %d / %d frames processed in %.1f seconds.' % \
                (thread_i, i + 1, slice_size, toc)
            print(msg, flush=True)
            tic = time.time()

def extract_motion(frames, reference_frame, filter_pct=99, n_threads=1):
    '''Extracts horizontal and vertical displacement field for each frame, from 
    the reference frame. This phase-based motion extraction uses
    complex steerable pyramids with n_scales=1 and n_orients=2.
    Args:
        frames -- list of images.
        reference_frame -- reference frame, usually first frame.
        filter_pct -- for each frame, any displacement above this percentile
            is removed.
        n_threads -- number of threads to use.
    Outputs:
        motion_fields (2, T, H, W) -- horizontal and vertical displacement
            for each frame.
    '''
    tic = time.time()
    T = len(frames)
    h, w = frames[0].shape
    # Pre-allocate memory for the numpy arrays.
    motion_fields = np.zeros((2, T, h, w))

    # Determine the chunk size of each thread.
    chunk_sizes = utils.get_chunk_sizes(T, n_threads)

    threads = []
    slice_start = 0
    print('Running motion extraction in %d threads...' % n_threads, flush=True)
    for thread_i in range(n_threads):
        slice_size = chunk_sizes[thread_i]
        slice_idxs = slice(slice_start, slice_start + slice_size)
        slice_frames = frames[slice_idxs]

        x = threading.Thread(
            target=_extract_motion_slice,
            args=(motion_fields, slice_frames, slice_idxs,
                  reference_frame, filter_pct, thread_i)
        )
        threads.append(x)
        x.start()

        # Update slice start.
        slice_start += slice_size

    for thread in threads:
        thread.join()

    elapsed_time = str(datetime.timedelta(seconds=time.time() - tic))
    print('Done! Elapsed time: %s' % elapsed_time)
    return motion_fields

def _weighted_gaussian_smooth_frame(image, amplitude_mask, sigma):
    num = gaussian_filter(image * amplitude_mask, sigma)
    den = gaussian_filter(amplitude_mask, sigma)
    smooth_image = num / den
    return smooth_image

def _weighted_gaussian_smooth_slice(motion_fields, slice_idxs,
                                    amp_mask_dx, amp_mask_dy, sigma, thread_i):
    verbose = thread_i == 0
    slice_size = slice_idxs.stop - slice_idxs.start

    tic = time.time()
    for i in range(slice_idxs.start, slice_idxs.stop):
        smooth_dx = _weighted_gaussian_smooth_frame(
            motion_fields[0, i], amp_mask_dx, sigma)
        smooth_dy = _weighted_gaussian_smooth_frame(
            motion_fields[1, i], amp_mask_dy, sigma)

        motion_fields[0, i, :, :] = smooth_dx
        motion_fields[1, i, :, :] = smooth_dy

        if verbose and (i + 1) % 500 == 0:
            toc = time.time() - tic
            msg = '[Thread %d] %d / %d frames processed in %.1f seconds.' % \
                (thread_i, i + 1, slice_size, toc)
            print(msg, flush=True)
            tic = time.time()

def weighted_gaussian_smooth(motion_fields, reference_frame, sigma=3, n_threads=1):
    '''Apply Gaussian kernel weighted by amplitude. 
    his will modify the images in-place.
    Args:
        motion_fields (2, T, H, W) -- list of motion field images.
        reference_frame -- reference image used to create amplitude masks.
        sigma -- std. dev. of Gaussian kernel, in pixels.
        n_threads -- number of threads to use.
    Output:
        smooth_motion_fields -- list of smoothed motion field images.
    '''
    tic = time.time()

    # Get amplitude mask (depending on horizontal or vertical direction).
    pyr = pt.pyramids.SteerablePyramidFreq(
        reference_frame, order=1, is_complex=True)
    Ax2 = np.abs(pyr.pyr_coeffs[(0,0)])**2
    Ay2 = np.abs(pyr.pyr_coeffs[(0,1)])**2

    # Determine the chunk size of each thread.
    T = motion_fields.shape[1]
    chunk_sizes = utils.get_chunk_sizes(T, n_threads)

    # Apply Gaussian blur to each frame.
    print('Running amplitude-weighted Gaussian smoothing in %d threads...' \
        % n_threads, flush=True)
    threads = []
    slice_start = 0
    for thread_i in range(n_threads):
        slice_size = chunk_sizes[thread_i]
        slice_idxs = slice(slice_start, slice_start + slice_size)

        x = threading.Thread(
            target=_weighted_gaussian_smooth_slice,
            args=(motion_fields, slice_idxs,
                  Ax2, Ay2, sigma, thread_i)
        )
        threads.append(x)
        x.start()

        # Update slice start.
        slice_start += slice_size

    for thread in threads:
        thread.join()

    elapsed_time = str(datetime.timedelta(seconds=time.time() - tic))
    print('Done! Elapsed time: %s' % elapsed_time)

    return motion_fields