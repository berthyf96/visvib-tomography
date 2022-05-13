import datetime
import os
import threading
import pickle
import time

import numpy as np
import pyrtools as pt
import cv2
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from cube import Cube
from solver import Solver


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

def plot_motion_field(motion_im, cmap_min=None, cmap_max=None,
                      cbar=True, fig=None, ax=None, **kwargs):
    '''Plot image of 2d displacement field (in seismic colormap).
    Args:
        motion_im -- 2d image of displacement field. 
        cmap_min -- min. value for colormap.
        cmap_max -- max. value for colormap.
        cbar -- whether to display the colobar.
        fig -- matplotlib figure on which to plot.
        ax -- matplotlib axis on which to plot.
    '''
    if cmap_min is None:
        cmap_min = -abs(motion_im).max()
    if cmap_max is None:
        cmap_max = abs(motion_im).max()
    norm = Normalize(-abs(motion_im).max(), abs(motion_im).max())
    if ax is not None:
        assert fig is not None
        im = ax.imshow(motion_im, norm=norm, cmap='seismic', **kwargs)
        if cbar:
            fig.colorbar(im, ax=ax)
        ax.axis('off')
    else:
        plt.imshow(motion_im, norm=norm, cmap='seismic', **kwargs)
        if cbar:
            plt.colorbar()
        plt.axis('off')
    return

def get_power_spectrum(fft):
    '''Return the power spectrum of the given FFT, where
    power = norm2(fft_coeffs)^2 for each FFT frequency bin.
    Args:
        fft -- ndarray, where axis 0 corresponds to the frequency bins.
    Output:
        spectrum -- ndarray of size len(fft), containing the power of each FFT
            frequency bin.
    '''
    n_freq_bins = len(fft)
    spectrum = np.zeros(n_freq_bins)
    for f in range(n_freq_bins):
        spectrum[f] = np.linalg.norm(fft[f])**2
    return spectrum

def find_peak_idxs(sig, **kwargs):
    '''Find indices of peaks in max-normalized 1d signal.
    Args:
        sig -- signal.
        **kwargs -- additional arguments passed to scipy.signal.find_peaks.
    '''
    peak_idxs, _ = find_peaks(sig / sig.max(), **kwargs)
    return peak_idxs

def find_nearest(arr, val):
    '''Return index of nearest item to val in arr.'''
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx

def prompt_clicks(image, clicks, **kwargs):
    '''Prompts user for cube keypoints on given image, and records
    coordinates of mouse clicks. Make sure to run
    `%matplotlib notebook` before calling this function in a notebook. 
    Order: bot left -> bot mid -> bot right
           -> top left back -> top left front -> top right back
           -> top right front
    Inputs:
        image -- image to show the user.
        clicks -- array that will be modified with additional mouse clicks.
        kwargs -- additional arguments for plt.imshow(image).
    '''
    # Record user clicks.
    def on_click(event):
        # pylint: disable=no-member
        clicks.append([event.xdata, event.ydata])

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.imshow(image, **kwargs)
    plt.show()

def keypoints_from_clicks(clicks, nx, ny, nz):
    '''Interpolates coordinates of visible mesh vertices given
    coordinates of keypoints in an image.
    '''
    p1, p2, p3, p4, p5, p6, p7 = np.asarray(clicks)[:]
    points = []
    # Going across the two vertical planes:
    for eta_z in np.linspace(0, 1, nz+1)[:-1]:
        ps = (1-eta_z)*p1 + eta_z*p4
        pt = (1-eta_z)*p2 + eta_z*p5
        for eta in np.linspace(0, 1, nx+1):
            points.append((1-eta)*ps + eta*pt)
        ps = (1-eta_z)*p2 + eta_z*p5
        pt = (1-eta_z)*p3 + eta_z*p7
        for eta in np.linspace(0, 1, ny+1)[1:]:
            points.append((1-eta)*ps + eta*pt)
    # Top plane:
    for eta_y in np.linspace(0, 1, ny+1):
        ps = (1-eta_y)*p4 + eta_y*p6
        pt = (1-eta_y)*p5 + eta_y*p7
        for eta in np.linspace(0, 1, nx+1):
            points.append((1-eta)*ps + eta*pt)
    points = np.array(points)
    return points

def get_cube_model(xlen, ylen, zlen, nx, ny, nz, nu, degree=1):
    '''Returns a Cube instance, assuming E = 1, rho = 1, nu = nu.'''
    n_vox = nx * ny * nz
    if np.isscalar(nu):
        elem_nus = np.ones(n_vox) * nu
    else:
        elem_nus = nu
    cube = Cube(
        xlen, ylen, zlen, nx, ny, nz,
        elem_Es=np.ones(n_vox), elem_rhos=np.ones(n_vox),
        elem_nus=elem_nus, deg=degree)
    return cube

def gather_image_space_modes(fft_dx, fft_dy, fft_idxs, is_complex=False):
    '''Returns arrays of shape (N_MODES, H, W) of
    horizontal mode images and vertical mode images.
    Args:
        fft_dx (N_FFT_BINS, H, W) -- FFT coeffs of horizontal motion.
        fft_dy (N_FFT_BINS, H, W) -- FFT coeffs of vertical motion.
        fft_idxs -- FFT indices that correspond to modes.
        is_complex -- whether to keep image-space modes complex-valued.
    '''
    _, H, W = fft_dx.shape
    if is_complex:
        modes_dx = np.zeros((len(fft_idxs), H, W), dtype=complex)
        modes_dy = np.zeros((len(fft_idxs), H, W), dtype=complex)
    else:
        modes_dx = np.zeros((len(fft_idxs), H, W))
        modes_dy = np.zeros((len(fft_idxs), H, W))
    for (i, fft_idx) in enumerate(fft_idxs):
        dx, dy = fft_dx[fft_idx], fft_dy[fft_idx]
        if not is_complex:
            dx = dx.real
            dy = dy.real
        modes_dx[i] = dx
        modes_dy[i] = dy
    return modes_dx, modes_dy

def sample_image_space_mode(image_space_mode_dx, image_space_mode_dy,
                            image_space_points):
    '''Sample one image-space mode at given points.
    Inputs:
        image_space_mode_dx (H, W) -- image-space mode of horizontal motion
        iamge_space_mode_dy (H, W) -- image-space mode of vertical motion
        image_space_points (N_PTS, 2) -- points at which
            to sample the image-space mode.
    Output:
        sampled_mode (N_PTS, 2) -- image-space mode sampled at given points.
    '''
    n_pts = len(image_space_points)
    sampled_mode = np.zeros((n_pts, 2), dtype=image_space_mode_dx.dtype)
    for (i, coords) in enumerate(image_space_points):
        x, y = int(coords[0]), int(coords[1])
        sampled_mode[i, 0] = image_space_mode_dx[y, x]
        sampled_mode[i, 1] = image_space_mode_dy[y, x]
    return sampled_mode

def get_observed_modal_data(image_space_modes_dx, image_space_modes_dy,
                            image_space_points, n_dofs, image_space_dofs):
    '''Sample observed image-space modes at given points and return
    a mode matrix U with the correct observed DOF ordering.
    Args:
        image_space_modes_dx (N_MODES, H, W) -- image-space modes of
            horizontal motion
        iamge_space_modes_dy (N_MODES, H, W) -- image-space modes of
            vertical motion
        image_space_points (N_PTS, 2) -- points at which
            to sample the image-space modes.
        n_dofs -- total number of DOFs.
        image_space_dofs (N_PTS * 3) -- DOFs in the observed image-space order.
    Outputs:
        U_observed (N_DOFS, N_MODES) -- all image-space modes sampled
            at given points.
    '''
    assert image_space_modes_dx.shape == image_space_modes_dy.shape
    n_modes = len(image_space_modes_dx)

    U_observed = np.zeros((n_dofs, n_modes), dtype=image_space_modes_dx.dtype)
    for i in range(n_modes):
        sampled_mode = sample_image_space_mode(
            image_space_modes_dx[i], image_space_modes_dy[i],
            image_space_points)

        # Fill DOFs corresponding to x direction.
        U_observed[image_space_dofs[::3], i] = sampled_mode[:, 0]

        # Fill DOFs corresponding to y direction.
        U_observed[image_space_dofs[1::3], i] = sampled_mode[:, 1]
    
    return U_observed

def normalize_modes(modes):
    '''Normalize modes by L2 norm.
    Inputs:
        modes -- 2d array of size (N_DOF, N_MODES).
    '''
    modes_normalized = modes.copy()
    for i in range(modes.shape[1]):
        modes_normalized[:, i] /= np.linalg.norm(modes_normalized[:, i])
    return modes_normalized

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

def projection_matrix_from_keypoints(cube, keypoints, flip_y=True):
    '''Estimate projection matrix mapping seen mesh vertices to given keypoints
    in the image.
    Args:
        cube -- Cube object, which contains coordinates of the seen mesh points.
        keypoints (N_KEYPOINTS, 2) -- image-space coordinates of user-selected
            keypoints.
        flip_y -- whether to flip vertical motion from image-space to
            world-space.
    Output:
        proj_mat (2, 3) -- 3d-to-2d projection matrix estimated from data.
    '''
    seen_mesh_points = \
        cube.V.tabulate_dof_coordinates()[cube.image_space_dofs[::3]]
    X = seen_mesh_points.T
    Y = keypoints.T.copy()
    Y[0] += -Y[0,0]
    Y[1] += -Y[1,0]
    proj_mat = Y @ X.T @ np.linalg.inv(X @ X.T)
    if flip_y:
        # Flip second row of projection matrix so that vertical 
        # motion is flipped.
        proj_mat[1] = -proj_mat[1]

    return proj_mat

def _get_chunk_sizes(n_items, n_threads):
    '''Given n_items to process in n_threads, returns the number of items to
    send to each chunk. This is useful for functions that use multithreading.
    '''
    min_chunk_size = n_items // n_threads
    if n_items % n_threads == 0:
        chunk_sizes = [min_chunk_size] * n_threads
    else:
        chunk_sizes = [min_chunk_size] * (n_threads - 1)
        remainder = n_items % n_threads
        chunk_sizes.append(min_chunk_size + remainder)
    return chunk_sizes

def eigvals_to_freqs(eigvals):
    '''Convert eigenvalues to frequencies [Hz].'''
    return np.sqrt(eigvals) / (2*np.pi)

def freqs_to_eigvals(freqs):
    '''Convert frequencies [Hz] to eigenvalues.'''
    return np.square(2*np.pi*freqs)

def full_mode_matrix(modes_free, n_total_dofs, free_dofs):
    '''Returns full mode matrix containing free and boundary DOFs.
    Inputs:
        modes_free (N_FREE_DOFS, N_MODES) -- mode matrix for only free DOFs.
        n_total_dofs -- number of total DOFs (including boundary DOFs).
        free_dofs -- free DOFs.
    Output:
        modes_full (N_TOTAL_DOFS, N_MODES) -- full mode matrix for all DOFs.
    '''
    n_modes = modes_free.shape[1]
    modes_full = np.zeros((n_total_dofs, n_modes), dtype=modes_free.dtype)
    modes_full[free_dofs] = modes_free
    return modes_full

def weighted_sum(arrays, weights, is_coo=True):
    '''Returns the weighted sum of given arrays.
    Inputs:
        arrays -- ndarray of size (N, ...) containing N ndarrays to combine.
        weights -- ndarray of size (N) containing the scalar weights.
        is_coo -- whether arrays are coo_matrix objects. If True, result
            will be of type scipy.sparse.csr.csr_matrix.
    Output:
        res -- weighted sum of size (...)
    '''
    if len(arrays) != len(weights):
        raise ValueError('len(arrays) != len(weights)')
    if len(weights[0].shape) > 0:
        raise ValueError('Weights are not scalars! Perhaps you swapped ' \
            'the argument order.')

    if is_coo:
        res = coo_matrix(arrays[0].shape, dtype=arrays[0].dtype)
    else:
        res = np.zeros(arrays[0].shape, dtype=arrays[0].dtype)
    for (wi, Ai) in zip(weights, arrays):
        res += wi * Ai
    return res

def project_points(pts, proj_matrix):
    '''Project 3d points onto 2d space, given projection matrix.
    Inputs:
        pts -- 3d points, an array of size (N_PTS, 3)
        proj_matrix -- projection matrix, an array of size (2, 3)
    '''
    projected_points = proj_matrix @ pts.T
    return projected_points.T

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
    motion_2d = project_points(motion_3d, proj_mat)

    # Interpolate 2d motion.
    interp_dx, interp_dy = interpolate_2d_motion(points, motion_2d)
    return interp_dx, interp_dy

def plot_3d_cube(cube, weights, wmin=None, wmax=None, cmap=cm.viridis,
                 title='', pad_height=5, elev=5, azim=None, width=3, height=12,
                 fig=None, ax=None, title_fontsize=21, cbar_fontsize=12,
                 cbar=True, cbar_vmin_loc=(0.5, -0.06), dpi=80):
    '''Plots the expanded 3D view of cube weights.'''
    if not wmin:
        wmin = weights.min()
    if not wmax:
        wmax = weights.max()
    norm = Normalize(wmin, wmax)

    # Plot each layer in the cube.
    voxels = np.zeros((cube.nx, cube.ny, pad_height*(cube.nz-1)+1))
    colors = np.zeros((cube.nx, cube.ny, pad_height*(cube.nz-1)+1, 4))
    for z in range(cube.nz):
        voxels[:, :, z*pad_height] = 1
        layer_weights = cube.layer_weights(weights, z)
        layer_weights = rotate(layer_weights, 90, reshape=True)
        layer_weights = np.flip(layer_weights, axis=0)
        layer_colors = cmap(norm(layer_weights))
        layer_colors = layer_colors.reshape(cube.nx, cube.ny, 4)
        colors[:, :, z*pad_height] = layer_colors

    if not fig and not ax:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        ax = Axes3D(fig, proj_type='ortho')
    
    if azim:
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=elev)
    ax.voxels(voxels, facecolors=colors)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    ax.axis('off')
    fig.suptitle(title, fontsize=title_fontsize)
    plt.axis('off')
    
    # Make colorbar.
    if cbar:
        clb = plt.colorbar(m, shrink=0.5, aspect=60)
        clb.ax.set_title('%d' % wmax, fontsize=cbar_fontsize)  # max value
        clb.ax.text(
            cbar_vmin_loc[0], cbar_vmin_loc[1],
            '%d' % wmin, fontsize=cbar_fontsize, ha='center', 
            transform=clb.ax.transAxes)  # min value
        clb.set_ticks([])
    
    return fig

# -----------------------------------------------------------------------------
# MOTION EXTRACTION UTILS
# -----------------------------------------------------------------------------

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
    chunk_sizes = _get_chunk_sizes(T, n_threads)

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
    chunk_sizes = _get_chunk_sizes(T, n_threads)

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

# -----------------------------------------------------------------------------
# MOTION FFT UTILS
# -----------------------------------------------------------------------------
def _fft_results_slice(fft_u, fft_v, motion_fields, slice_idxs, thread_i):
    verbose = thread_i == 0
    slice_size = slice_idxs.stop - slice_idxs.start
    width = motion_fields.shape[-1]
    n_fft_bins = len(fft_u)

    tic = time.time()

    for row in range(slice_idxs.start, slice_idxs.stop):
        for col in range(width):
            pixel_fft_u = np.fft.fft(motion_fields[0, :, row, col], axis=0)
            pixel_fft_v = np.fft.fft(motion_fields[1, :, row, col], axis=0)

            fft_u[:, row, col] = pixel_fft_u[:n_fft_bins]
            fft_v[:, row, col] = pixel_fft_v[:n_fft_bins]

        if verbose and (row + 1) % 5 == 0:
            toc = time.time() - tic
            msg = '[Thread %d] %d / %d rows processed in %.1f seconds.' % \
                (thread_i, row + 1, slice_size, toc)
            print(msg, flush=True)
            tic = time.time()

def get_fft_results(motion_fields, sample_rate, max_fft_freq=None, n_threads=1):
    '''Performs 1d FFT of motion for each pixel.
    Inputs:
        motion_fields (2, T, H, W) -- horizontal + vertical displacement
            of each pixel in each frame.
        sample_rate -- FPS of video.
        max_fft_freq -- maximum FFT frequency bin to include.
        n_threads -- number of threads to use.
    Outputs:
        fft_u (T/2, H, W): FFT of horizontal motion for each pixel.
        fft_v (T/2, H, W): FFT of vertical motion for each pixel.
        power_spectrum (T/2): squared power for each frequency.
        ffreqs (T/2): FFT frequencies (up to Nyquist).
    '''
    tic = time.time()
    _, T, height, width = motion_fields.shape

    # Get FFT frequencies and max-frequency index.
    ffreqs = np.fft.fftfreq(T) * sample_rate
    nyq = T // 2  # Nyquist index
    if not max_fft_freq:
        n_fft_bins = nyq
    else:
        n_fft_bins = find_nearest(ffreqs, max_fft_freq)

    # Pre-allocate memory.
    fft_u = np.zeros((n_fft_bins, height, width), dtype=complex)
    fft_v = np.zeros((n_fft_bins, height, width), dtype=complex)

    # Each thread will process a chunk of rows.
    chunk_sizes = _get_chunk_sizes(height, n_threads)

    print('Running FFT in %d threads...' % n_threads, flush=True)
    threads = []
    slice_start = 0
    for thread_i in range(n_threads):
        slice_size = chunk_sizes[thread_i]
        slice_idxs = slice(slice_start, slice_start + slice_size)

        x = threading.Thread(
            target=_fft_results_slice,
            args=(fft_u, fft_v, motion_fields, slice_idxs, thread_i)
        )
        threads.append(x)
        x.start()

        # Update slice start.
        slice_start += slice_size

    for thread in threads:
        thread.join()

    # Compute power spectrum.
    power_spectrum_u = get_power_spectrum(fft_u)
    power_spectrum_v = get_power_spectrum(fft_v)
    power_spectrum = power_spectrum_u + power_spectrum_v

    elapsed_time = str(datetime.timedelta(seconds=time.time() - tic))
    print('Done! Elapsed time: %s' % elapsed_time)

    return fft_u, fft_v, power_spectrum, ffreqs[:n_fft_bins]


# -----------------------------------------------------------------------------
# SOLVER UTILS
# -----------------------------------------------------------------------------
def laplacian_matrix_3d(nx, ny, nz):
    '''Return a Laplacian matrix (L) for a 3d array.
    L[i, j] = 1 iff i != j are adjacent in the 3d array of size
    (nx, ny, nz).
    L[i, i] = -deg(i).
    '''
    def _index(x, y, z):
        return (nx*ny)*z + nx*y + x
    nelems = nx * ny * nz
    L = np.zeros((nelems, nelems))
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                curr = _index(x, y, z)
                if x > 0:
                    up = _index(x-1, y, z)
                    L[curr, up] = 1
                if x < nx - 1:
                    down = _index(x+1, y, z)
                    L[curr, down] = 1
                if y > 0:
                    left = _index(x, y-1, z)
                    L[curr, left] = 1
                if y < ny - 1:
                    right = _index(x, y+1, z)
                    L[curr, right] = 1
                if z > 0:
                    bottom = _index(x, y, z-1)
                    L[curr, bottom] = 1
                if z < nz - 1:
                    top = _index(x, y, z+1)
                    L[curr, top] = 1
                L[curr, curr]  = -np.count_nonzero(L[curr])
    return L

def full_projection_matrix(n_total_pts, proj_mat):
    '''Create a full projection matrix that maps all DOFs onto image-space.
    Inputs:
        n_total_pts -- total number of points that would be mapped from
            3d to 2d.
        proj_mat (2, 3) -- 3d-to-2d projection matrix.
    Output:
        P (N_DOFS, N_DOFS) -- full projection matrix, that maps DOFs in 3d
            to 2d image-space, where the 3rd dimension is always 0.
    '''
    P = np.kron(
        np.eye(n_total_pts, dtype=int),
        np.r_[proj_mat, np.zeros((1, 3))])
    return P

def full_mask_matrix(n_dofs, seen_dofs):
    '''Return a binary matrix with 1s for DOFs seen in image-space.'''
    G = np.zeros((n_dofs, n_dofs))
    G[seen_dofs[::3], seen_dofs[::3]] = 1
    G[seen_dofs[1::3], seen_dofs[1::3]] = 1
    return G

def get_solver(cube, proj_mat):
    '''Initializes a Solver for the given Cube and projection matrix.'''
    # Laplacian operator
    L = laplacian_matrix_3d(cube.nx, cube.ny, cube.nz)

    # voxel matrices
    n_voxels = len(cube.element_stiffness_mats)
    element_stiffness_mats, element_mass_mats = [], []
    for (Ke, Me) in tqdm(zip(cube.element_stiffness_mats,
                             cube.element_mass_mats), total=n_voxels,
                            desc='Gather element mats'):
        Ke_free = Ke.tocsr()[cube.nonbc_dofs].tocsc()[:, cube.nonbc_dofs].tocoo()
        Me_free = Me.tocsr()[cube.nonbc_dofs].tocsc()[:, cube.nonbc_dofs].tocoo()
        element_stiffness_mats.append(Ke_free.astype('float32'))
        element_mass_mats.append(Me_free.astype('float32'))

    # projection matrices
    n_vts = cube.n_dofs // 3
    proj_mat_normalized = proj_mat / abs(proj_mat).max()
    P = full_projection_matrix(n_vts, proj_mat_normalized)
    G = full_mask_matrix(cube.n_dofs, cube.image_space_dofs)
    P_free = P[cube.nonbc_dofs][:, cube.nonbc_dofs]
    G_free = G[cube.nonbc_dofs][:, cube.nonbc_dofs]

    # Initialize solver.
    s = Solver(
        element_stiffness_mats, element_mass_mats, 
        laplacian=L, P=P_free, G=G_free)
    
    return s


def gather_modal_data_across_videos(info_dict_fns):
    '''Returns all observed (normalized, real) modes and frequencies, as well as other FFT data.
    Input:
        info_dict_fns -- list of filenames containing pickle dictionaries.
    Outputs:
        data_dict (dict) -- dictionary containing all sampled 3D modes, 
            image-space modes, etc.
    '''
    print('Gathering modal observation data from:')
    for fn in info_dict_fns:
        print('  * %s' % fn)

    obs_modes_dx, obs_modes_dy, obs_freqs = [], [], []
    obs_modes_sampled = []
    obs_modes_powers = []
    fft_bin_widths = []
    for info_dict_fn in info_dict_fns:
        with open(info_dict_fn, 'rb') as fp:
            info_dict = pickle.load(fp)

        modes_dx = info_dict['modes_dx'].real
        modes_dy = info_dict['modes_dy'].real
        freqs = info_dict['freqs']
        fft_res = info_dict['fft_res']
        powers = info_dict['powers']
        U_observed = np.real(info_dict['U_observed'])

        # Take the real part of and normalize each mode.
        for i, (dx, dy) in enumerate(zip(modes_dx, modes_dy)):
            norm = np.linalg.norm(np.concatenate((dx.real, dy.real)))
            modes_dx[i] = np.real(dx) / norm
            modes_dy[i] = np.real(dy) / norm
        for i in range(U_observed.shape[-1]):
            U_observed[:, i] = U_observed[:, i] / np.linalg.norm(U_observed[:, i])

        obs_modes_sampled.append(U_observed)
        obs_modes_dx.append(modes_dx)
        obs_modes_dy.append(modes_dy)
        obs_freqs.append(freqs)
        obs_modes_powers.append(powers)
        fft_bin_widths.append(fft_res)

    # Process image-space modes: keep real part only and flip images vertically.
    for vid_i in range(len(obs_modes_dx)):
        for mode_i in range(len(obs_modes_dx[vid_i])):
            dx = obs_modes_dx[vid_i][mode_i].real
            dy = obs_modes_dy[vid_i][mode_i].real
            obs_modes_dx[vid_i][mode_i] = cv2.flip(dx, 0)
            obs_modes_dy[vid_i][mode_i] = cv2.flip(dy, 0)

    data_dict = {
        'obs_3d_modes': obs_modes_sampled,
        'obs_modes_dx': obs_modes_dx,
        'obs_modes_dy': obs_modes_dy,
        'obs_freqs': obs_freqs,
        'obs_modes_powers': obs_modes_powers,
        'fft_bin_widths': fft_bin_widths
    }
    return data_dict