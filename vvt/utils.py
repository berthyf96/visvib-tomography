import io

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix

def get_chunk_sizes(n_items, n_threads):
    """
    Determine the number of items to send to each chunk. This is used when
    chunking data in multithreaded functions.

    Parameters
    ----------
    n_items: int
        Total number of items that need to be processed. E.g., if there are
        1000 frames that need to processed by 5 threads, then `n_items`
        would be 1000.
    n_threads: int
        Total number of threads that will be used.

    Returns
    -------
    chunk_sizes: list of int
        The number of items to be sent to each thread. The last thread 
        could get a number that's smaller than that of the other threads.
    """
    min_chunk_size = n_items // n_threads
    if n_items % n_threads == 0:
        chunk_sizes = [min_chunk_size] * n_threads
    else:
        chunk_sizes = [min_chunk_size] * (n_threads - 1)
        remainder = n_items % n_threads
        chunk_sizes.append(min_chunk_size + remainder)
    return chunk_sizes

def find_nearest(arr, val):
    """
    Return the index of the nearest item to `val` in `arr`.
    """
    arr = np.asarray(arr)
    idx = (np.abs(arr - val)).argmin()
    return idx

def eigvals_to_freqs(eigvals):
    """
    Convert eigenvalues to frequencies [Hz].
    """
    return np.sqrt(eigvals) / (2*np.pi)

def freqs_to_eigvals(freqs):
    """
    Convert frequencies [Hz] to eigenvalues.
    """
    return np.square(2*np.pi*freqs)

def full_mode_matrix(modes_free, n_total_dofs, free_dofs):
    """
    Construct a full mode matrix containing free and boundary DOFs.

    Parameters
    -----------
    modes_free: np.ndarray of shape (N_FREE_MODES, N_MODES)
        The full-field mode matrix for only the free DOFs. This is the
        matrix `U` that is a decision variable in the optimization procedure.
    n_total_dofs: int
        Total number of DOFs (including boundary DOFs).
    free_dofs: np.array or list
        The list of free DOFs.

    Returns
    -------
    modes_full: np.ndarray of shape (N_TOTAL_DOFS, N_MODES)
        Full mode matrix for all DOFs.
    """
    n_modes = modes_free.shape[1]
    modes_full = np.zeros((n_total_dofs, n_modes), dtype=modes_free.dtype)
    modes_full[free_dofs] = modes_free
    return modes_full

def prompt_clicks(image, clicks, **kwargs):
    """
    Prompt user for the pixel locations of reference mesh points on the cube.
    Make sure to run `%matplotlib notebook` before calling this function in 
    a Jupyter notebook. The order of the clicks needs to be:
        bot left -> bot mid -> bot right
        -> top left back -> top left front -> top right back
        -> top right front
    
    Parameters
    ----------
    image: np.ndarray
        2D image to show the user.
    clicks: list
        The empty list that will contain the locations of the mouse clicks.
    **kwargs: variable keyword arguments
        Additional keyword arguments for `plt.imshow(image)`.
    """
    # Record user clicks.
    def on_click(event):
        # pylint: disable=no-member
        clicks.append([event.xdata, event.ydata])

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.imshow(image, **kwargs)
    plt.show()

def keypoints_from_clicks(clicks, nx, ny, nz):
    """
    Interpolate the coordinates of all visible mesh vertices given the 
    coordinates of reference keypoints in the image.

    Parameters
    ----------
    clicks: np.array or list
        The user-clicked locations of the reference mesh vertices in the
        image.
    nx, ny, nz: int
        The number of voxels, or hexahedra, in the x, y, and z directions of
        the cube model.

    Returns
    -------
    keypoints: np.ndarray of shape (N_VISIBLE_POINTS, 2)
        The interpolated pixel coordinates of all the visible mesh vertices.
    """
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
    """
    Estimate the projection matrix mapping seen mesh vertices, or keypoints,
    from their 3D coordinates to their 2D image-space coordinates.

    Parameters
    ----------
    cube: Cube
        Cube object, which contains the 3D coordinates of the seen mesh points.
    keypoints: np.ndarray of shape (N_VISIBLE_PTS, 2)
        The pixel coordinates of the visible mesh points.
    flip_y: bool, default=True
        Whether to flip vertical motion from image-space to world-space.

    Returns
    -------
    proj_mat: np.ndarray of shape (2, 3)
        3D-to-2D projection matrix estimated from the data.
    """
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
    """
    Compute the weighted sum of the given arrays.

    Parameters
    ----------
    arrays: np.ndarray or scipy.sparse matrix of shape (N, ...)
        The N arrays to combine.
    weights: np.ndarray of shape (N)
        The scalar weight of each array.
    is_coo: bool, default=True
        Whether the arrays are scipy.sparse.coo_matrix objects. If True,
        then the result will be of type scipy.sparse.csr.csr_matrix.

    Returns
    -------
    res: np.ndarray or scipy.sparse.csr.csr_matrix
        The weighted sum of the arrays.
    """
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
    """
    Return the 3D points projected onto 2D space.
    """
    projected_points = proj_matrix @ pts.T
    return projected_points.T

def interp_3d_data(data_points, data_values, query_points):
    """
    Interpolate a 3D function and its values at the query points.

    Parameters
    ----------
    data_points: np.ndarray of shape (N_PTS, 3)
        3D coordinates of the sampled known points.
    data_values: np.ndarray of shape (N_PTS, DIM)
        The values at the sampled points.
    query_points: np.ndarray of shape (N_QUERY_PTS, 3)
        The points at which to query the interpolated function.

    Returns
    -------
    yhat: np.ndarray of shape (N_QUERY_PTS, DIM)
        The interpolated values at the query points.
    """
    interp = LinearNDInterpolator(
        (data_points[:, 0], data_points[:, 1], data_points[:, 2]),
        data_values)
    yhat = interp(query_points[:, 0], query_points[:, 1], query_points[:, 2])
    yhat[np.isnan(yhat)] = 0
    return yhat

def fig_to_im(fig, dpi=180, is_grayscale=False, transparent=False):
    """
    Return an np.ndarray of the matplotlib figure.
    Source: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """
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
    """
    Writes a GIF of the given list of images to the specified output file.
    """
    imageio.mimsave(out_fn, images, **kwargs)
    return