import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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