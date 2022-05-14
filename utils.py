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

from scipy.sparse import coo_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from cube import Cube
from vvt.solver import Solver



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