import datetime
import pickle
import threading
import time

import cv2
import numpy as np
import scipy.signal

import vvt.utils as utils

def get_power_spectrum(fft):
    """
    Compute the power spectrum of the given FFT, where the power is defined as
        power = norm2(fft_coeffs)^2
    for each FFT frequency bin.

    Parameters
    ----------
    fft: ndarray
        The FFT coefficients, where axis 0 corresponds to the frequency bins.

    Returns
    -------
    spectrum: ndarray
        A 1D numpy array of the power at each FFT frequency bin.
    """
    n_freq_bins = len(fft)
    spectrum = np.zeros(n_freq_bins)
    for f in range(n_freq_bins):
        spectrum[f] = np.linalg.norm(fft[f])**2
    return spectrum

def find_peak_idxs(signal, **kwargs):
    """
    Fing the indices of peaks in the max-normalized 1D signal.

    Parameters
    ----------
    signal: np.ndarray
        A 1D numpy array containing the input signal.
    **kwargs: variable arguments
        Optional additional arguments passed to `scipy.signal.find_peaks`.

    Returns
    -------
    peak_idxs: np.ndarray
        A 1D numpy array of the indices (if any) of `signal` that
        correspond to peaks.
    """
    peak_idxs, _ = scipy.signal.find_peaks(signal / signal.max(), **kwargs)
    return peak_idxs

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
    """
    Perform 1D FFT of motion for each pixel.

    Parameters
    ----------
    motion_fields: np.ndarray of shape (2, T, H, W)
        The horizontal and vertical displacement of each pixel in each frame.
    sample_rate: int
        The sampling rate (in FPS) of the video.
    max_fft_freq: float, optional
        The maximum FFT frequency to include in the results. By default, all
        frequencies up to the Nyquist sampling rate (T/2) are included.
    n_threads: int, default=1
        Number of threads to use for multithreading.

    Returns
    -------
    fft_u, fft_v: ndarray of shape (N_FFT_BINS, H, W)
        The FFT of horizontal motion and vertical motion, respectively, for
        each pixel. N_FFT_BINS = T/2, unless a lower `max_fft_freq` is given.
    ffreqs: ndarray of shape (N_FFT_BINS)
        The FFT frequencies.
    """
    tic = time.time()
    _, T, height, width = motion_fields.shape

    # Get FFT frequencies and max-frequency index.
    ffreqs = np.fft.fftfreq(T) * sample_rate
    nyq = T // 2  # Nyquist index
    if not max_fft_freq:
        n_fft_bins = nyq
    else:
        n_fft_bins = utils.find_nearest(ffreqs, max_fft_freq)

    # Pre-allocate memory.
    fft_u = np.zeros((n_fft_bins, height, width), dtype=complex)
    fft_v = np.zeros((n_fft_bins, height, width), dtype=complex)

    # Each thread will process a chunk of rows.
    chunk_sizes = utils.get_chunk_sizes(height, n_threads)

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

    return fft_u, fft_v, power_spectrum, ffreqs
    
def gather_image_space_modes(fft_u, fft_v, fft_idxs, is_complex=False):
    """
    Extract the image-space modes corresponding to the given FFT bins.

    Parameters
    ----------
    fft_u, fft_v: np.ndarray of shape (N_FFT_BINS, H, W)
        The FFT of horizontal motion and vertical motion, respectively, for
        each pixel.
    fft_idxs: np.ndarray or list
        A list of 0-indexed FFT indices (i.e., bins) corresponding to modes.
    is_complex: bool, default=False
        Whether to keep the image-space modes complex-valued.

    Returns
    -------
    modes_dx, modes_dy: np.ndarray of shape (N_MODES, H, W)
        2D numpy arrays corresponding to images of the image-space modes
        (in the horizontal and vertical directions, respectively).
    """
    _, H, W = fft_u.shape
    if is_complex:
        modes_dx = np.zeros((len(fft_idxs), H, W), dtype=complex)
        modes_dy = np.zeros((len(fft_idxs), H, W), dtype=complex)
    else:
        modes_dx = np.zeros((len(fft_idxs), H, W))
        modes_dy = np.zeros((len(fft_idxs), H, W))
    for (i, fft_idx) in enumerate(fft_idxs):
        dx, dy = fft_u[fft_idx], fft_v[fft_idx]
        if not is_complex:
            dx = dx.real
            dy = dy.real
        modes_dx[i] = dx
        modes_dy[i] = dy
    return modes_dx, modes_dy

def sample_image_space_mode(image_space_mode_dx, image_space_mode_dy,
                            image_space_points):
    """
    Sample an image-space mode at the given points.

    Parameters
    ----------
    image_space_mode_dx, image_space_mode_dy: np.ndarray of shape (H, W)
        The images of horizontal and vertical motion, respectively, of the
        image-space mode.
    image_space_points: np.ndarray of shape (N_PTS, 2)
        Pixel coordinates of the points at which to sample the image-space mode.

    Returns
    -------
    sampled_mode: np.ndarray of shape (N_PTS, 2)
        The image-space mode sampled at the specified points.
    """
    n_pts = len(image_space_points)
    sampled_mode = np.zeros((n_pts, 2), dtype=image_space_mode_dx.dtype)
    for (i, coords) in enumerate(image_space_points):
        x, y = int(coords[0]), int(coords[1])
        sampled_mode[i, 0] = image_space_mode_dx[y, x]
        sampled_mode[i, 1] = image_space_mode_dy[y, x]
    return sampled_mode

def get_observed_modal_data(image_space_modes_dx, image_space_modes_dy,
                            image_space_points, n_dofs, image_space_dofs):
    """
    Sample image-space modes at the the given points, and construct a 
    mode matrix U whose DOFs that are visible in image-space are filled-in
    with their modal motion in image-space.

    Parameters
    ----------
    image_space_mode_dx, image_space_mode_dy: np.ndarray of shape (N_MODES, H, W)
        The images of horizontal and vertical motion, respectively, of the
        image-space modes to be sampled.
    image_space_points: np.ndarray of shape (N_PTS, 2)
        Pixel coordinates of the points at which to sample the image-space mode.
    n_dofs: int
        The total number of DOFs.
    image_space_dofs: np.ndarray of shape (N_PTS * 3)
        The image-space DOFs in the observed image-space order.

    Returns
    -------
    U_observed: np.ndarray of shape (`n_dofs`, N_MODES)
        The full-field mode matrix containing the image-space modes sampled
        at the given points. Any DOFs not visible in image-space are set to 0.
    """
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

def normalize_modes(modes):
    """
    Normalize modes by the L2 norm.

    Parameters
    ----------
    modes: np.ndarray of shape (N_DOFS, N_MODES)
        The mode matrix to be normalized.

    Returns
    -------
    modes_normalized: np.ndarray of shape (N_DOFS, N_MODES)
        The normalized mode matrix, i.e., each column has L2 norm = 1.
    """
    modes_normalized = modes.copy()
    for i in range(modes.shape[1]):
        modes_normalized[:, i] /= np.linalg.norm(modes_normalized[:, i])
    return modes_normalized