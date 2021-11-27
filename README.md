# Visual Vibration Tomography
## Dependencies
* python 3.7
* conda install -c conda-forge fenics
* conda install -c conda-forge notebook
* conda install h5py
* conda install -c conda-forge matplotlib==3.4.2
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
utils.py
solver.py
real_data/
    cube.avi
    drum1.avi
simulated_data/
    defect1/
        defect1_top_front_pluck.gif
        true_stiffness.npy
        true_density.npy
comsol/
    template.mph
    run_comsol_sim.m
demo_simulated_cube.ipynb
demo_real_cube.ipynb
demo_real_drum.ipynb
```

# TODOs
* Add true weights to `simulated_data` and compare in `demo_simulated_cube.ipynb`.