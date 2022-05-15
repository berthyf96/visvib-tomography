# Visual Vibration Tomography
## Setup
### Create conda environment
```
$ conda update conda
$ conda create -n vvt python=3.7
$ conda activate vvt
```
### Install dependencies
```
$ conda config --add channels conda-forge
$ conda install -c conda-forge fenics
$ conda install -c conda-forge jupyter matplotlib==3.4.2 tqdm opencv imageio
$ conda install -c anaconda scipy
$ pip install pyrtools
```
* python 3.7
* conda install -c conda-forge fenics
* conda install -c conda-forge notebook
* conda install h5py
<!-- * conda install -c conda-forge matplotlib==3.4.2 -->
* conda install -c conda-forge matplotlib==2.2.4
* conda install -c conda-forge tqdm
* conda install -c conda-forge opencv
* conda install -c anaconda scipy
* conda install -c conda-forge imageio
* pip install pyrtools

## Demos
### Simulated Cube
* COMSOL transient analysis
* animation creation
* motion extraction + inference

### Real Cube
* motion extraction + inference

### Real Drum
* motion extraction + inference

## Structure
```
real_data/
    defect_cube/
        top_front_pluck.avi
        top_right_pluck.avi
        left_side_twist.avi
simulated_data/
    defect01/
        defect1_top_front_pluck.gif
        true_stiffness.npy
        true_density.npy
comsol/
    template.mph
    run_comsol_sim.m
scripts/
    comsol_animation.py
demo_simulated_cube.ipynb
demo_real_cube.ipynb
demo_real_drum.ipynb
```

# TODOs
* Upload simulated data to Dropbox / Google Drive
* damped simulation demo