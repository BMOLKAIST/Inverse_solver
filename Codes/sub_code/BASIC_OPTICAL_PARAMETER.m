function params= BASIC_OPTICAL_PARAMETER(init_params)
params=struct;
params.size=[100 100 100];%size of the refractive index
params.wavelength=0.532;%wavelength
params.NA=1.2;%objective lens NA
params.RI_bg=1.336;%refractive index out of the sample
params.resolution=[0.1 0.1 0.1];%resolution of one voxel
params.vector_simulation=true;%use polarised field or scalar field
params.use_abbe_sine=true;

if nargin==1
    params=update_struct(params,init_params);
end

end