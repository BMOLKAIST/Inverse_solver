clc, clear;
cd0 = matlab.desktop.editor.getActiveFilename;
dash = cd0(strfind(cd0,'EXPERIMENT_BEAD.m')-1);
cd0 = cd0(1:strfind(cd0,'EXPERIMENT_BEAD.m')-2);
addpath(genpath(cd0));
%% set the experimental parameters
cddata = [cd0 dash 'Data'];
bg_file = [cddata dash 'SiO2_1_bg.tif'];
sp_file =  [cddata dash 'SiO2_1_sp.tif'];

cd(cddata)
odt = load('SIM_spec.mat').odt;

%1 optical parameters
MULTI_GPU=false;

params=BASIC_OPTICAL_PARAMETER();
params.NA=1.16;
params.RI_bg=odt.n_m;
params.wavelength=odt.lambda;
params.resolution=[1 1 1]*params.wavelength/4/params.NA;
params.vector_simulation=false;true;
params.size=[0 0 71]; 
params.use_GPU = true;

%2 illumination parameters
field_retrieval_params=FIELD_EXPERIMENTAL_RETRIEVAL.get_default_parameters(params);
field_retrieval_params.resolution_image=[1 1]*(odt.pix/odt.mag);
field_retrieval_params.conjugate_field=true;
field_retrieval_params.use_abbe_correction=true;

% 1. Aberration correction
field_retrieval=FIELD_EXPERIMENTAL_RETRIEVAL(field_retrieval_params);

% Aberration correction data
[input_field,field_trans,params]=field_retrieval.get_fields(bg_file,sp_file);

% Display results: transmitted field
figure;orthosliceViewer(squeeze(abs(field_trans(:,:,:)./input_field(:,:,:))),'displayrange',[0 2]); colormap gray; title('Amplitude')
figure;orthosliceViewer(squeeze(angle(field_trans(:,:,:)./input_field(:,:,:)))); colormap jet; title('Phase')


%% Set forward parameters
%4 forward solver parameters
forward_params=FORWARD_SOLVER_CONVERGENT_BORN.get_default_parameters(params);
forward_params.use_GPU=true;
%5 multiple scattering solver
if ~MULTI_GPU
    backward_params=BACKWARD_SOLVER_MULTI.get_default_parameters(params);
else
    backward_params=BACKWARD_SOLVER_MULTI_MULTI_GPU.get_default_parameters(params);
end


forward_params_backward=FORWARD_SOLVER_CONVERGENT_BORN.get_default_parameters(forward_params);
forward_params_backward.return_transmission=true;
forward_params_backward.return_reflection=true;
forward_params_backward.return_3D=true;
forward_params_backward.boundary_thickness=[2 2 4]; % if xy is nonzero, acyclic convolution is applied.
%forward_solver_backward=FORWARD_SOLVER_CONVERGENT_BORN(forward_params_backward);
backward_params.forward_solver=@(x) FORWARD_SOLVER_CONVERGENT_BORN(x);%forward_solver_backward;
backward_params.forward_solver_parameters=forward_params_backward;


%% solve the backward multiple scattering problem
init_backward_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
init_solver_backward=BACKWARD_SOLVER_RYTOV(init_backward_params);
backward_params.init_solver=init_solver_backward;

% Parameters
backward_params.vector_simulation = false;
backward_params.verbose = true;
backward_params.use_abbe_sine = true;
backward_params.use_non_negativity = true;
backward_params.nmax = inf;
backward_params.nmin = params.RI_bg;
backward_params.itter_max=200;
backward_params.inner_itt=400;
backward_params.step = 2.5e-3;
tv_param_list = 7.5e-2;
selected_field = unique([1 randperm(size(input_field,4),40)]);
selected_field = selected_field(1:40);
    
    
Error_ours = inf;
for j1 = 1:length(tv_param_list)
    backward_params.tv_param = tv_param_list(j1);
    backward_params.num_scan_per_iteration = 0;
    if ~MULTI_GPU
        backward_solver=BACKWARD_SOLVER_MULTI(backward_params);
    else
        backward_solver=BACKWARD_SOLVER_MULTI_MULTI_GPU(backward_params);
    end
    t1 = clock;
    RI_multi=backward_solver.solve(input_field(:,:,:,selected_field),field_trans(:,:,:,selected_field));
    elapse_time = etime(clock,t1);
    tv_param = tv_param_list(j1);
    backward_solver.delete
end
backward_params.tv_param = tv_param;

%% solve with rytov
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
RI_rytov=rytov_solver.solve(input_field,field_trans);
figure;orthosliceViewer(real(RI_rytov)); title('Rytov')
%% solve with rytov + TV
backward_single_params=BACKWARD_SOLVER_SINGLE.get_default_parameters(params);
init_backward_single_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
init_solver_backward_single=BACKWARD_SOLVER_RYTOV(init_backward_single_params);
backward_single_params.init_solver=init_solver_backward_single;

backward_single_params.itter_max=200;
backward_single_params.inner_itt=400;
backward_single_params.forward_solver=@(x) FORWARD_SOLVER_CONVERGENT_BORN(x);%forward_solver_backward;
backward_single_params.forward_solver_parameters=forward_params_backward;

backward_single_params.vector_simulation = false;
backward_single_params.verbose = false;
backward_single_params.use_abbe_sine = true;
backward_single_params.use_non_negativity = true;
backward_single_params.nmax = inf;
backward_single_params.nmin = params.RI_bg;
backward_single_params.step = 5e-3;
tv_param_list = 7.5e-2;

for j1 = 1:length(tv_param_list)
    backward_single_params.tv_param = tv_param_list(j1);
    single_solver=BACKWARD_SOLVER_SINGLE(backward_single_params);
    t1 = clock;
    RI_single=single_solver.solve(input_field,field_trans);
    tv_param = tv_param_list(j1);
    elapse_time = etime(clock,t1);
end
backward_single_params.tv_param = tv_param;
figure;orthosliceViewer(real(RI_single)); title('Rytov + TV')


