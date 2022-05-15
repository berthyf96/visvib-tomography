import io

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix

def get_chunk_sizes(n_items, n_threads):
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

def find_nearest(arr, val):
    '''Return index of nearest item to val in arr.'''
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx

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

def interp_3d_data(data_points, data_values, query_points):
    '''Interpolate 3D function and return query values.
    Inputs:
        data_points (N_points, 3) -- sampled known points.
        data_values (N_points, N_dim) -- values of sampled points.
        query_points (N_query_points, 3) -- requested interpolated points.
    '''
    interp = LinearNDInterpolator(
        (data_points[:, 0], data_points[:, 1], data_points[:, 2]),
        data_values)
    yhat = interp(query_points[:, 0], query_points[:, 1], query_points[:, 2])
    yhat[np.isnan(yhat)] = 0
    return yhat

def fig_to_im(fig, dpi=180, is_grayscale=False, transparent=False):
    '''Returns an ndarray of the matplotlib figure image.
    Source: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    '''
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=transparent)
    buf.seek(0)
    im_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    if is_grayscale:
        im = cv2.imdecode(im_arr, 0)
    else:
        im = cv2.imdecode(im_arr, 1)
    return im

def ims_to_gif(out_fn, images, **kwargs):
    '''Writes GIFs of given list of images to specified output file.'''
    imageio.mimsave(out_fn, images, **kwargs)
    return