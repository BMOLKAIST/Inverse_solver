%{
field_file='Q:\shin_cell_phantom\shin_cell_phantom\shin_pol_setup_cell_phantom_data.mat';
load(field_file);
%}
%%
%wrapped_phase=single(gpuArray(retPhase));
%wrapped_phase=100*single(gpuArray(rand(500,500,500)));

%close all
%%
%{
bg =single( (loadTIFF('C:\Users\Administrator\Desktop\HERVE\__MULTIPLE_scattering\datas\raw_bg.tiff')));
bg=bg(:,:,1:100);
sp =single( (loadTIFF('C:\Users\Administrator\Desktop\HERVE\__MULTIPLE_scattering\datas\raw_tomo78.tiff')));
sp=sp(:,:,1:100);
%}
%%
%{
imsize=size(bg,1);
lambda=0.457; % wavelength
pixel_size=5.5;
M=55.555;
NA=0.75;
res1 = pixel_size/M;%(=camera pixel size in um);
% spatial frequency resolution by field of view
kres1 = 1/(imsize*pixel_size/M);%(=Fourier space pixel size in um^(-1));
NAlimit_radius = NA/lambda/kres1;
NA_mask =  single(~mk_ellipse(NAlimit_radius, NAlimit_radius, imsize, imsize));

field=QPI_GPU(sp,bg,NA_mask,NAlimit_radius);

wrapped_phase=angle(single(gpuArray(field)));

%}
%%

wait(gpuDevice());tic; 
%for ii=1:100
residue_map = get_residue_gpu(wrapped_phase);
[residue_map,step1_unwrapp,step2_unwrapp]=remove_dipoles_gpu(residue_map,int32(find(residue_map(:)>0))); %also need the position of positive residues
[step1_unwrapp,step2_unwrapp]=goldstein_branch_cut(residue_map,step1_unwrapp,step2_unwrapp);
unwrapped_phase=unwrapp_phase_goldstein(wrapped_phase,step1_unwrapp,step2_unwrapp);
%end
wait(gpuDevice());toc;


[residue_idx,residue_value] = get_residu_position_values(residue_map);
%%
%{
cpu_wrapped_phase=gather(wrapped_phase);

tic;
for ii=1: size(cpu_wrapped_phase,3)
    cpu_wrapped_phase(:,:,ii)=single(unwrap2(double(cpu_wrapped_phase(:,:,ii))));
end
toc;
%}
