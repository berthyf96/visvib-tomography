import numpy as np
import scipy.ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_motion_field(motion_im, vmin=None, vmax=None,
                      cbar=True, fig=None, ax=None, **kwargs):
    """
    Plot image of 2D displacement field (in seismic colormap).

    Parameters
    ----------
    motion_im: np.ndarray
        A 2D image of the displacement field.
    vmin, vmax: float, optional
        The min. and max. values of the seismic colormap. If not specified,
        colormap will be centered at 0 and cover the maximum displacement
        magnitude.
    cbar: bool, default=True
        Whether to plot the colorbar alongside the displacement-field image.
    fig, ax: mpl.figure.Figure and mpl.axes._subplots.AxesSubplot, optional
        A specific matplotlib figure container and subplot axis on which to
        plot the image. These must be specified together, or not at all.
    """
    if vmin is None:
        vmin = -np.max(abs(motion_im))
    if vmax is None:
        vmax = np.max(abs(motion_im))
    norm = mpl.colors.Normalize(vmin, vmax)
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

def plot_3d_cube(cube, weights, vmin=None, vmax=None, cmap='viridis',
                 pad_height=5, elev=25, azim=0, fig=None, ax=None,
                 cbar=True, cbar_fontsize=12, cbar_vmin_loc=(0.5, -0.06), **fig_kwargs):
    """
    Plot an expanded 3D view of cube weights (i.e., material-property values).

    Parameters
    ----------
    cube: Cube
        The Cube instance. This needs to be provided to know how to assign
        the voxel-wise weights.
    weights: np.array or list
        The list of voxel-wise material-property values. These weights will
        be assigned to voxels accordingly.
    vmin, vmax: float, optional
        The range of values covered by the colormap. If not specified,
        colormap will cover the entire range of values in `weights`.
    cmap: str, default='viridis'
        A string identifier for the desired colormap.
    pad_height: int, default=5
        The amount of vertical space between layers of the cube. Specifically,
        the number of voxels to skip in the z direction between layers.
    elev: float, default=20
        The elevation of the camera viewing the 3D axis.
    azim: float, default=0
        The azimuthal angle (in degrees) of the camera viewing the 3D axis.
    fig, ax: mpl.figure.Figure and mpl.axes._subplots.AxesSubplot, optional
        A specific matplotlib figure container and subplot axis on which to
        plot the image.
    cbar: bool, default=True
        Whether to plot the colorbar alongside the cube.
    cbar_fontsize: int, default=12
        The fontsize of the colorbar labels. Only the min. and max. values
        are labeled.
    cbar_vmin_loc: tuple, default=(0.5, -0.06)
        The location of the min. value label of the colorbar.
    **fig_kwargs: variable arguments
        Variable arguments for creating a new mpl.figure.Figure (when one is
        not specified).
    """
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.min(weights)
    if vmax is None:
        vmax = np.max(weights)
    norm = mpl.colors.Normalize(vmin, vmax)
    
    # Plot each layer in the cube.
    voxels = np.zeros((cube.nx, cube.ny, pad_height * (cube.nz - 1) + 1))
    colors = np.zeros((cube.nx, cube.ny, pad_height * (cube.nz - 1) + 1, 4))
    for z in range(cube.nz):
        voxels[:, :, z * pad_height] = 1
        layer_weights = cube.layer_weights(weights, z)
        layer_weights = scipy.ndimage.rotate(layer_weights, 90, reshape=True)
        layer_weights = np.flip(layer_weights, axis=0)
        layer_colors = cmap(norm(layer_weights))
        layer_colors = layer_colors.reshape(cube.nx, cube.ny, 4)
        colors[:, :, z*pad_height] = layer_colors

    if fig is None:
        fig = plt.figure(**fig_kwargs)
    if ax is None:
        ax = Axes3D(fig, proj_type='ortho')
    
    if azim:
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=elev)
    ax.voxels(voxels, facecolors=colors)
    m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    ax.axis('off')
    ax.set_box_aspect((1, 1, 6))
    
    # Make colorbar.
    if cbar:
        clb = plt.colorbar(m, shrink=0.5, aspect=60)
        clb.ax.set_title('%d' % vmax, fontsize=cbar_fontsize)  # max value
        clb.ax.text(
            cbar_vmin_loc[0], cbar_vmin_loc[1],
            '%d' % vmin, fontsize=cbar_fontsize, ha='center', 
            transform=clb.ax.transAxes)  # min value
        clb.set_ticks([])
    
    return fig