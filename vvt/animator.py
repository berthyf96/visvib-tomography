import datetime
import os
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import vvt.utils as utils

class Animator:
    def __init__(self, nodes, us, vs, ws, subtract_bottom=True,
                 grid_res=100, n_rand_pts=3000):
        """
        Initialize Animator object.

        Parameters
        ----------
        nodes: ndarray of shape (N_NODES, 3)
            The (x, y, z) coordinates of each mesh vertex, or node.
        us, vs, ws: ndarrays of shape (N_FRAMES, N_NODES)
            The node displacements in the x, y, and z directions, respectively,
            for each frame.
        subtract_bottom: bool, default=True
            Whether to subtract the motion of the bottom nodes.
        grid_res: int, default=100
            The resolution of the 3D interpolation grid that is used to
            compute the displacements of the random scatter points.
        n_rand_pts: int, default=3000
            The number of random points to plot on the surface of the cube.
            These points will provide a speckle-like texture for motion extraction.
        """
        self.nodes = nodes
        self.n_pts = len(nodes)
        self.n_frames = len(us)
        self.xmin, self.xmax = nodes[:,0].min(), nodes[:,0].max()
        self.ymin, self.ymax = nodes[:,1].min(), nodes[:,1].max()
        self.zmin, self.zmax = nodes[:,2].min(), nodes[:,2].max()

        def _inplane1(pt):
            return np.isclose(pt[1], self.ymin)
        
        def _inplane2(pt):
            return np.isclose(pt[0], self.xmax)
        
        def _inplane3(pt):
            return np.isclose(pt[2], self.zmax)
        
        def _inview(pt):
            return _inplane1(pt) or _inplane2(pt) or _inplane3(pt)
        
        def _bottom(pt):
            return np.isclose(pt[2], self.zmin)

        # Get nodes that are in view.
        self.view_idxs = [i for (i, pt) in enumerate(nodes) if _inview(pt)]

        # Get bottom nodes and free nodes.
        self.bottom_idxs = [i for (i, pt) in enumerate(nodes) if _bottom(pt)]
        self.free_idxs = [i for (i, pt) in enumerate(nodes) if not _bottom(pt)]
        self.free_view_idxs = [i for (i, pt) in enumerate(nodes) \
            if _inview(pt) and not _bottom(pt)]
        self.bottom_view_idxs = [i for (i, pt) in enumerate(nodes) \
            if _inview(pt) and _bottom(pt)]

        if subtract_bottom:
            self.us, self.vs, self.ws = self.get_bottom_subtracted_displacements(
                us, vs, ws)
        else:
            self.us, self.vs, self.ws = us, vs, ws

        # Make a 3D grid of higher resolution.
        x = np.linspace(self.xmin, self.xmax, grid_res)
        y = np.linspace(self.ymin, self.ymax, grid_res)
        z = np.linspace(self.zmin, self.zmax, grid_res)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
        self.grid_x, self.grid_y, self.grid_z = grid_x, grid_y, grid_z

        # Choose random points on the grid to plot.
        grid_points = np.c_[
            grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]
        
        # TODO: Make this step faster.
        grid_view_idxs = [i \
            for (i, pt) in enumerate(grid_points) if _inview(pt)]

        random_idxs = np.random.choice(
            grid_view_idxs, n_rand_pts, replace=False)
        self.n_rand_pts = n_rand_pts
        self.random_idxs = random_idxs
        self.plot_points = grid_points[random_idxs]

    def _interp_displacements_slice(self, interp_disps, slice_idxs,
                                    start_frame, thread_i):
        verbose = thread_i == 0
        slice_size = slice_idxs.stop - slice_idxs.start

        tic = time.time()
        for i in range(slice_size):
            t = start_frame + slice_idxs.start + i

            u, v, w = self.us[t], self.vs[t], self.ws[t]
            uvw = np.c_[u, v, w]

            interp_uvw = utils.interp_3d_data(
                self.nodes, uvw, self.plot_points)
            interp_disps[t] = interp_uvw

            if verbose and (i + 1) % 25 == 0:
                toc = time.time() - tic
                msg = '[Thread %d] %d / %d frames processed in %.1f seconds.' % \
                (thread_i, i + 1, slice_size, toc)
                print(msg, flush=True)
                tic = time.time()

    def interp_displacements(self, start_frame=0, end_frame=None, n_threads=1):
        """
        For each frame, interpolate the displacements of the random
        scatter points based on the displacements of the mesh nodes.
        """
        if not end_frame:
            end_frame = self.n_frames
        n_frames = end_frame - start_frame

        # Pre-allocate space.
        interp_disps = np.zeros((n_frames, self.n_rand_pts, 3))

        # Determine the chunk size of each tread.
        chunk_sizes = utils.get_chunk_sizes(n_frames, n_threads)
        
        print('Starting %d threads...' % n_threads, flush=True)
        threads = []
        slice_start = 0
        tic = time.time()
        for thread_i in range(n_threads):
            slice_size = chunk_sizes[thread_i]
            slice_idxs = slice(slice_start, slice_start + slice_size)

            x = threading.Thread(
                target=self._interp_displacements_slice,
                args=(interp_disps, slice_idxs, start_frame, thread_i)
            )
            threads.append(x)
            x.start()

            # Update slice start.
            slice_start += slice_size

        for thread in threads:
            thread.join()

        elapsed_time = str(datetime.timedelta(seconds=time.time() - tic))
        print('Done! Elapsed time: %s' % elapsed_time)
        
        return interp_disps

    def get_bottom_subtracted_displacements(self, us, vs, ws):
        """
        Subtract the motion of the bottom nodes from the displacements of all
        the nodes.
        """
        bottom_us = np.mean(us[:, self.bottom_idxs], axis=1)
        bottom_vs = np.mean(vs[:, self.bottom_idxs], axis=1)
        bottom_ws = np.mean(ws[:, self.bottom_idxs], axis=1)
        
        sub_us = us - np.tile(bottom_us, (self.n_pts, 1)).T
        sub_vs = vs - np.tile(bottom_vs, (self.n_pts, 1)).T
        sub_ws = ws - np.tile(bottom_ws, (self.n_pts, 1)).T
        
        return sub_us, sub_vs, sub_ws

    def update_displacements(self, us, vs, ws, subtract_bottom=True):
        if subtract_bottom:
            self.us, self.vs, self.ws = self.get_bottom_subtracted_displacements(
                us, vs, ws)
        else:
            self.us, self.vs, self.ws = us, vs, ws

    def plot_cube(self, points, xrange=None, yrange=None, zrange=None,
                  figsize=(8, 8), dpi=72):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.view_init(azim=-50)
        plt.axis('off')

        # Plot scatter points.
        m = len(points) // 2
        ax.scatter(
            points[:m, 0], points[:m, 1], points[:m, 2], s=1,
            c='orange')
        ax.scatter(
            points[m:, 0], points[m:, 1], points[m:, 2], s=1,
            c='black')

        if xrange:
            ax.set_xlim(xrange[0], xrange[1])
        if yrange:
            ax.set_ylim(yrange[0], yrange[1])
        if zrange:
            ax.set_zlim(zrange[0], zrange[1])

        # Make tight layout.
        fig.subplots_adjust(.02, .02, .98, .98)
        
        return fig, ax

    def render_frames(self, interp_disps, scaling=1,
                      figsize=(8, 8), dpi=72):
        """
        Render frames as scatter plots of random surface points moving.
        """
        # Get range of positions for plotting.
        xrange = (self.xmin - abs(scaling * self.us).max(),
                  self.xmax + abs(scaling * self.us).max())
        yrange = (self.ymin - abs(scaling * self.vs).max(),
                  self.ymax + abs(scaling * self.vs).max())
        zrange = (self.zmin - abs(scaling * self.ws).max(),
                  self.zmax + abs(scaling * self.ws).max())

        n_frames = len(interp_disps)
        height = figsize[0] * dpi
        width = figsize[1] * dpi
        
        # Pre-allocate space.
        frames = np.zeros((n_frames, height, width))

        for t in tqdm(range(n_frames), desc='Render frames'):
            disp = interp_disps[t] * scaling
            pos = self.plot_points + disp
            fig, _ = self.plot_cube(
                pos, xrange, yrange, zrange, figsize, dpi)
            frame = utils.fig_to_im(fig, dpi=dpi, is_grayscale=True)
            frames[t] = frame
            plt.close()
    
        return frames.astype(np.uint8)

    def write_frames_to_dir(self, dir, interp_disps, scaling=1,
                      figsize=(8, 8), dpi=72):
        """
        Render frames of the moving scatter points, and save the images to the
        given directory, `dir`.
        """
        # Get range of positions for plotting.
        xrange = (self.xmin - abs(scaling * self.us).max(),
                  self.xmax + abs(scaling * self.us).max())
        yrange = (self.ymin - abs(scaling * self.vs).max(),
                  self.ymax + abs(scaling * self.vs).max())
        zrange = (self.zmin - abs(scaling * self.ws).max(),
                  self.zmax + abs(scaling * self.ws).max())

        n_frames = len(interp_disps)

        for t in tqdm(range(n_frames), desc='Render frames'):
            disp = interp_disps[t] * scaling
            pos = self.plot_points + disp
            fig, _ = self.plot_cube(
                pos, xrange, yrange, zrange, figsize, dpi)
            fig.savefig(os.path.join(dir, 'frame_%d.png' % t), transparent=True)
            plt.close()
    
        return

    def get_proj_mat(self):
        _, ax = self.plot_cube(self.nodes)
        plt.show(block=False)
        proj_mat = ax.M
        plt.close()
        return proj_mat

    def get_projected_displacements(self, proj_mat=None, scaling=1):
        if proj_mat is None:
            proj_mat = self.get_proj_mat(scaling=scaling)
            proj_mat = proj_mat[:2, :3]

        # Project true displacements onto image-space.
        image_space_displacements = []
        for (u, v, w) in tqdm(zip(self.us, self.vs, self.ws), total=len(self.us),
                              desc='Project displacements'):
            disps = np.column_stack((u, v, w))
            projected_disps = utils.project_points(disps, proj_mat)
            image_space_displacements.append(projected_disps)
        image_space_displacements = np.stack(image_space_displacements)
        
        return image_space_displacements

    def get_projected_nodes(self, proj_mat=None, scaling=1.):
        if proj_mat is None:
            proj_mat = self.get_proj_mat(scaling=scaling)
            proj_mat = proj_mat[:2, :3]
        
        # Project 3D node coordinates onto image-space.
        projected_nodes = utils.project_points(self.nodes, proj_mat)
        
        return projected_nodes