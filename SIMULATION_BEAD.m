clc, clear;
cd0 = matlab.desktop.editor.getActiveFilename;
dash = cd0(strfind(cd0,'SIMULATION_BEAD.m')-1);
cd0 = cd0(1:strfind(cd0,'SIMULATION_BEAD.m')-2);
addpath(genpath(cd0));
used_gpu_device=1;
gpu_device=gpuDevice(used_gpu_device);
%% set the simulation parameters
MULTI_GPU=false; % Use Multiple GPU?

%1 optical parameters
params=BASIC_OPTICAL_PARAMETER();
params.NA=1.2; % Numerical aperture
params.RI_bg=1.336; % Background RI
params.wavelength=0.532; % [um]
params.resolution=[1 1 1]*params.wavelength/4/params.NA; % 3D Voxel size [um]
params.use_abbe_sine=true; % Abbe sine condition according to demagnification condition
params.vector_simulation=true;false; % True/false: dyadic/scalar Green's function
params.size=[161 161 61]; % 3D volume grid
%2 illumination parameters
field_generator_params=FIELD_GENERATOR.get_default_parameters(params);
field_generator_params.illumination_number=40; 
field_generator_params.illumination_style='random';%'circle';%'random';%'mesh'
%3 phantom generation parameter
phantom_params=PHANTOM.get_default_parameters();
phantom_params.name='bead';%'RBC';
RI_sp=1.4609;
phantom_params.outer_size=params.size;
phantom_params.inner_size=round(ones(1,3) * 5 ./ params.resolution);
phantom_params.rotation_angles = [0 0 0];


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
forward_params_backward.used_gpu = 0;
%6 parameter for rytov solver


%% create phantom and solve the forward problem
% make the phantom
RI=PHANTOM.get(phantom_params);
RI=params.RI_bg+RI.*(RI_sp-params.RI_bg);
%create the incident field
field_generator=FIELD_GENERATOR(field_generator_params);
input_field=field_generator.get_fields();
%compute the forward field - CBS
forward_solver=FORWARD_SOLVER_CONVERGENT_BORN(forward_params);
forward_solver.set_RI(RI);
tic;
[field_trans,field_ref,field_3D]=forward_solver.solve(input_field);
toc;

% Display results: transmitted field
[input_field_scalar,field_trans_scalar]=vector2scalarfield(input_field,field_trans);
input_field_no_zero=input_field_scalar;zero_part_mask=abs(input_field_no_zero)<=0.01.*mean(abs(input_field_no_zero(:)));input_field_no_zero(zero_part_mask)=0.01.*exp(1i.*angle(input_field_no_zero(zero_part_mask)));
figure;orthosliceViewer(squeeze(abs(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap gray; title('Amplitude')
figure;orthosliceViewer(squeeze(angle(field_trans_scalar(:,:,:)./input_field_no_zero(:,:,:)))); colormap jet; title('Phase')


%% solve the backward multiple scattering problem

backward_params.forward_solver=@(x) FORWARD_SOLVER_CONVERGENT_BORN(x);%forward_solver_backward;
backward_params.forward_solver_parameters=forward_params_backward;
init_backward_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
init_solver_backward=BACKWARD_SOLVER_RYTOV(init_backward_params);
backward_params.init_solver=init_solver_backward;

% Set parameters
backward_params.vector_simulation = false; % true - dyadic ; false - scalar
backward_params.verbose = true; % Draw figures
backward_params.use_abbe_sine = true;
backward_params.use_non_negativity = false;
backward_params.nmax = inf;
backward_params.itter_max=200;
backward_params.inner_itt=400;
tv_param_list = 7.5e-3;

% Run!
Error_ours = inf;
for j1 = 1:length(tv_param_list)
    backward_params.tv_param = tv_param_list(j1);
    backward_params.num_scan_per_iteration = 8; % Stochastic gradient field number; 0 -> full-field is used.
    if ~MULTI_GPU
        backward_solver=BACKWARD_SOLVER_MULTI(backward_params);
    else
        backward_solver=BACKWARD_SOLVER_MULTI_MULTI_GPU(backward_params);
    end
    t1 = clock;
    RI_multi2=backward_solver.solve(input_field,field_trans);
    elapse_time0 = etime(clock,t1);
    Error = sum(abs(RI_multi2-RI).^2,'all') / sum(abs(params.RI_bg-RI).^2,'all');
    if Error < Error_ours
        RI_multi = RI_multi2;
        tv_param = tv_param_list(j1);
        Error_ours = Error;
        elapse_time = elapse_time0;
    end
    backward_solver.delete
end
backward_params.tv_param = tv_param;

figure;orthosliceViewer(real(RI)); title('Phantom RI')
figure;orthosliceViewer(real(RI_multi)); title('Our method')
  %% solve with rytov
rytov_params=BACKWARD_SOLVER_RYTOV.get_default_parameters(params);
rytov_params.use_non_negativity=false;
rytov_solver=BACKWARD_SOLVER_RYTOV(rytov_params);
RI_rytov=rytov_solver.solve(input_field,field_trans);
Error_Rytov = sum(abs(real(RI_rytov)-RI).^2,'all') / sum(abs(params.RI_bg-RI).^2,'all');
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
backward_single_params.nmin = min(RI(:));


tv_param_list = 5e-2;

Error_single = inf;
for j1 = 1:length(tv_param_list)
    backward_single_params.tv_param = tv_param_list(j1);
    single_solver=BACKWARD_SOLVER_SINGLE(backward_single_params);

    t1 = clock;
    RI_single2=single_solver.solve(input_field,field_trans);
    elapse_time0 = etime(clock,t1);
    Error = sum(abs(RI_single2-RI).^2,'all') / sum(abs(params.RI_bg-RI).^2,'all');
    if Error < Error_single
        RI_single = RI_single2;
        tv_param = tv_param_list(j1);
        Error_single = Error;
        elapse_time = elapse_time0;
    end
end
backward_single_params.tv_param = tv_param;
figure;orthosliceViewer(real(RI_single)); title('Rytov + TV')
