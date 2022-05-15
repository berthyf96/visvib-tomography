"""
This script makes and saves an animation of a cube vibrating.
It starts with a `.mat` file containing the results of a COMSOL transient 
analysis, specifically the mesh node locations and 3D displacements over time.
It mimics a speckle pattern by choosing random scatter points on the cube's
surface, and it plots these scatter points at each time according to the
displacements given by the COMSOL results. The matplotlib figure at each
frame is saved as an image, and all the frames are saved as a GIF.
This script may take a long time (e.g., about half an hour for the default 
data and settings).

Example usage: `python make_comsol_animation.py defect03 top_front_pluck`
"""
import argparse
import os

import scipy.io

import vvt.animator
import vvt.utils

AMPLIFY_FACTOR = 0.2
N_RAND_POINTS = 5000
INTERP_RES = 50
DPI = 72
FIGSIZE = (7, 7)
START_FRAME, END_FRAME = 0, None
FPS = 30
N_THREADS = 8

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_name',
                        type=str,
                        help="object name (e.g., 'defect_cube_center')")
    parser.add_argument('sim_name',
                        type=str,
                        help="simulation name (e.g., 'top_front_pluck')")
    args = parser.parse_args()
    return args

def main(args):
    obj_name = args.obj_name
    sim_name = args.sim_name
    sim_dir = os.path.join('simulated_data', obj_name, sim_name)

    msg = '|   %s / %s   |' % (obj_name, sim_name)
    print('=' * len(msg), flush=True)
    print(msg, flush=True)
    print('=' * len(msg), flush=True)

    gif_fn = os.path.join(sim_dir, 'transient.gif')
    if os.path.exists(gif_fn):
        raise FileExistsError('File exists:', gif_fn)

    # Load COMSOL transient analysis results.
    comsol_results_fn = os.path.join(sim_dir, 'transient.mat')
    if not os.path.exists(comsol_results_fn):
        raise FileNotFoundError(
            'COMSOL transient analysis results not found:', comsol_results_fn)
    comsol_results = scipy.io.loadmat(comsol_results_fn)
    
    # Initialize Animator object.
    animator = vvt.animator.Animator(
        nodes=comsol_results['p'],  # node locations
        us=comsol_results['u'],     # node x-displacements
        vs=comsol_results['v'],     # node y-displacements
        ws=comsol_results['w'],     # node z-displacements
        subtract_bottom=False,
        grid_res=INTERP_RES,
        n_rand_pts=N_RAND_POINTS)
    
    # Get interpolated displacements for all plot points.
    interp_disps = animator.interp_displacements(START_FRAME, END_FRAME, N_THREADS)

    # Render frames.
    frames = animator.render_frames(interp_disps, AMPLIFY_FACTOR, FIGSIZE, DPI)

    # Write GIF.
    vvt.utils.ims_to_gif(gif_fn, frames, fps=FPS)
    print('Wrote:', gif_fn)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)