clear; close all;

import com.comsol.model.*
import com.comsol.model.util.*

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TEMPLATE_MPH_FILENAME = 'template.mph';
OBJ_NAME = 'defect_cube_center_undamped_linear_samenu_test';
DATA_DIR = '../comsol_data';

IS_LINEAR = true;  % element order

% damping parameters
% CRIT_DAMPING_FRAC1 = 0.01749;  % at 12.5 Hz
% CRIT_DAMPING_FRAC2 = 0.01999;  % at 15.5 Hz
CRIT_DAMPING_FRAC1 = 0.0;  % at 12.5 Hz
CRIT_DAMPING_FRAC2 = 0.0;  % at 15.5 Hz

% defect dimensions
DEFECT_SIZE_X = '4*a/10';
DEFECT_SIZE_Y = '4*a/10';
DEFECT_SIZE_Z = '4*a/10';

% defect location
DEFECT_OFFSET_X = '0*a/10';
DEFECT_OFFSET_Y = '0*a/10';
DEFECT_OFFSET_Z = '0*a/10';

% material properties
E_JELLO = 9000;
E_CLAY = 5e6;
RHO_JELLO = 1270;
RHO_CLAY = 7620;
NU_JELLO = 0.3;
NU_CLAY = 0.3;

% simulation parameters
FRAME_RATE = '2000 [Hz]';
T_FINAL = '6 [s]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MAX_CHUNK_SIZE = 2000;
FORCINGS = {'top_plane_push', 'top_right_pluck', 'top_front_pluck',...
    'top_right_twist', 'top_left_pluck', 'top_back_pluck',};
FORCING_DISP_IDS = {'disp2', 'disp3', 'disp4', 'disp5', 'disp6', 'disp7'};
DISABLED_PHYSICS = {'solid/disp2', 'solid/disp3', 'solid/disp4',...
    'solid/disp5', 'solid/disp6', 'solid/disp7'};
RUN_FORCING_IDXS = [3];  % indices of forcing types to simulate

TA_STUDY_ID = 'std1';
TA_DSET_ID = 'dset1';

MA_STUDY_ID = 'std2';
EIGVECS_DSET_ID = 'dset2';
EIGVALS_SOL_ID = 'sol2';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_total = tic;

% Create results folder.
objDir = fullfile(DATA_DIR, OBJ_NAME);
[~, ~, ~] = mkdir(objDir);

% Load model.
model = mphload(TEMPLATE_MPH_FILENAME);

% Set model parameters.
model.param.set('crit_damping_frac1', CRIT_DAMPING_FRAC1);
model.param.set('crit_damping_frac2', CRIT_DAMPING_FRAC2);
model.param.set('defect_size_x', DEFECT_SIZE_X);
model.param.set('defect_size_y', DEFECT_SIZE_Y);
model.param.set('defect_size_z', DEFECT_SIZE_Z);
model.param.set('defect_offset_x', DEFECT_OFFSET_X);
model.param.set('defect_offset_y', DEFECT_OFFSET_Y);
model.param.set('defect_offset_z', DEFECT_OFFSET_Z);
model.param.set('E_jello', E_JELLO);
model.param.set('E_clay', E_CLAY);
model.param.set('rho_jello', RHO_JELLO);
model.param.set('rho_clay', RHO_CLAY);
model.param.set('nu_jello', NU_JELLO);
model.param.set('nu_clay', NU_CLAY);
model.param.set('frame_rate', FRAME_RATE);
model.param.set('t_final', T_FINAL);

% Set element order.
if IS_LINEAR
    model.physics('solid').prop('ShapeProperty').set(...
        'order_displacement', '1');
else
    model.physics('solid').prop('ShapeProperty').set(...
        'order_displacement', '2s');
end

% Build geoometry/mesh.
model.geom('geom1').run;
model.mesh('mesh1').run;

% Get model parameters.
N_t = mphevaluate(model, 'N_t');
frame_rate = mphevaluate(model, 'frame_rate');
t_final = mphevaluate(model, 't_final');
crit_damping_frac1 = mphevaluate(model, 'crit_damping_frac1');
crit_damping_frac2 = mphevaluate(model, 'crit_damping_frac2');
damping_freq1 = mphevaluate(model, 'damping_freq1');
damping_freq2 = mphevaluate(model, 'damping_freq2');
E_jello = mphevaluate(model, 'E_jello');
E_clay = mphevaluate(model, 'E_clay');
nu_jello = mphevaluate(model, 'nu_jello');
nu_clay = mphevaluate(model, 'nu_clay');
rho_jello = mphevaluate(model, 'rho_jello');
rho_clay = mphevaluate(model, 'rho_clay');
cube_len = mphevaluate(model, 'a');
defect_size_x = mphevaluate(model, 'defect_size_x');
defect_size_y = mphevaluate(model, 'defect_size_y');
defect_size_z = mphevaluate(model, 'defect_size_z');
defect_offset_x = mphevaluate(model, 'defect_offset_x');
defect_offset_y = mphevaluate(model, 'defect_offset_y');
defect_offset_z = mphevaluate(model, 'defect_offset_z');

% Re-convert defect dimensions into voxel units.
defect_size_x_vox = (defect_size_x * 10) / cube_len;
defect_size_y_vox = (defect_size_y * 10) / cube_len;
defect_size_z_vox = (defect_size_z * 10) / cube_len;
defect_offset_x_vox = (defect_offset_x * 10) / cube_len;
defect_offset_y_vox = (defect_offset_y * 10) / cube_len;
defect_offset_z_vox = (defect_offset_z * 10) / cube_len;

% Print info.
msg = sprintf('|     Processing %s from template %s     |',...
    OBJ_NAME, TEMPLATE_MPH_FILENAME);
disp(repmat('=', 1, length(msg)))
disp(msg)
disp(repmat('=', 1, length(msg)))
disp(['crit_damping_frac1 = ' num2str(crit_damping_frac1)])
disp(['crit_damping_frac2 = ' num2str(crit_damping_frac2)])
disp(['defect_size_x [vox] = ' num2str(defect_size_x_vox)])
disp(['defect_size_y [vox] = ' num2str(defect_size_y_vox)])
disp(['defect_size_z [vox] = ' num2str(defect_size_z_vox)])
disp(['defect_offset_x [vox] = ' num2str(defect_offset_x_vox)])
disp(['defect_offset_y [vox] = ' num2str(defect_offset_y_vox)])
disp(['defect_offset_z [vox] = ' num2str(defect_offset_z_vox)])
disp(['E_jello = ' num2str(E_jello)])
disp(['E_clay = ' num2str(E_clay)])
disp(['rho_jello = ' num2str(rho_jello)])
disp(['rho_clay = ' num2str(rho_clay)])
disp(['nu_jello = ' num2str(nu_jello)])
disp(['nu_clay = ' num2str(nu_clay)])
disp(['frame_rate = ' num2str(frame_rate)])
disp(['t_final = ' num2str(t_final)])
disp(repmat('=', 1, length(msg)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TRANSIENT ANALYSIS

for i = 1:length(RUN_FORCING_IDXS)
    forcingIdx = RUN_FORCING_IDXS(i);
    forcingLabel = FORCINGS{forcingIdx};
    
    simDir = fullfile(DATA_DIR, OBJ_NAME, forcingLabel);
    [~, ~, ~] = mkdir(simDir);
    
    % Check if these results already exist.
    saveFn = fullfile(simDir, 'transient.mat');
    if isfile(saveFn)
        disp([saveFn ' already exists.'])
        continue
    end

    % Enable/disable prescribed displacements accordingly.
    for j = 1:length(FORCING_DISP_IDS)
        featureId = FORCING_DISP_IDS{j};
        if j == forcingIdx
            model.physics('solid').feature(featureId).active(true);
        else
            model.physics('solid').feature(featureId).active(false);
        end
    end

    % Ensure that forcing prescribed displacements are disabled.
    model.study(TA_STUDY_ID).feature('time').set('disabledphysics',...
        DISABLED_PHYSICS);

    % Perform transient analysis.
    fprintf('Performing transient analysis for %s... ', forcingLabel)
    tic
    model.study(TA_STUDY_ID).run;
    fprintf('Done! ')
    toc

    % Extract displacement solutions.
    fprintf('Extracting displacement solution for %s...\n', forcingLabel)
    tic
    [dat, p] = mpheval2(model, {'u','v','w'}, TA_DSET_ID, MAX_CHUNK_SIZE);
    u = dat{1};
    v = dat{2};
    w = dat{3};
    toc
    
    % Save results.
    fprintf('Saving displacement solution for %s... ', forcingLabel)
    tic
    save(saveFn, 'frame_rate', 'N_t', 't_final',...
        'cube_len', 'defect_size_x', 'defect_size_y', 'defect_size_z',...
        'defect_offset_x', 'defect_offset_y', 'defect_offset_z',...
        'crit_damping_frac1', 'crit_damping_frac2', 'damping_freq1',...
        'damping_freq2',...
        'E_jello', 'E_clay', 'nu_jello', 'nu_clay', 'rho_jello',...
        'rho_clay', 'IS_LINEAR',...
        'u', 'v', 'w', 'p');
    fprintf('Done! ')
    toc
    
    disp('===========================================================')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MODAL ANALYSIS

% Disable forcing physics for modal analysis.
model.study(MA_STUDY_ID).feature('eig').set('disabledphysics',...
    DISABLED_PHYSICS);

% Run modal analysis.
fprintf('Performing modal analysis... ')
tic
model.study(MA_STUDY_ID).run;
fprintf('Done! ')
toc

% Extract results.
si = mphsolinfo(model, 'solname', EIGVALS_SOL_ID);
eigfreqs = si.solvals/(-2*pi*1i);

eigvec_disp = mpheval(model, {'u','v','w'}, 'dataset', EIGVECS_DSET_ID);
eigvecs_u = eigvec_disp.d1';
eigvecs_v = eigvec_disp.d2';
eigvecs_w = eigvec_disp.d3';

% Save results.
saveFn = fullfile(objDir, 'modal.mat');
save(saveFn, 'cube_len', 'defect_size_x', 'defect_size_y',...
    'defect_size_z', 'defect_offset_x', 'defect_offset_y',...
    'defect_offset_z', 'crit_damping_frac1', 'crit_damping_frac2',...
    'damping_freq1', 'damping_freq2',...
    'E_jello', 'E_clay', 'nu_jello', 'nu_clay', 'rho_jello',...
    'rho_clay', 'eigfreqs', 'eigvecs_u', 'eigvecs_v', 'eigvecs_w',...
    'IS_LINEAR');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ModelUtil.remove('Model');
fprintf('All done! ')
toc(t_total)