import datetime
import threading
import time

import numpy as np
import scipy

import vvt.utils as utils

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
    peak_idxs, _ = scipy.signal.find_peaks(sig / sig.max(), **kwargs)
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