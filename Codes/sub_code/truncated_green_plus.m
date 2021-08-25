function green_out=truncated_green_plus(params,refocus_version)
%refocus version has no k_z so it refocuses not green function
if ~exist('refocus_version','var')
    refocus_version=false;
end
%inspired from : "Fast convolution with free-space Green’s functions" Felipe Vico Leslie Greengard Miguel Ferrando
%check that input has all the requires params
params_required=BASIC_OPTICAL_PARAMETER();
params_required.use_GPU=true;
params_required.oversize_z=10;
params_required.oversample_xy=20000;%200;
params_required.oversample_xy_fourier=2;
params_required.simultanous_process=30;
%params_required
%params
params=update_struct(params_required,params);%check for reuired parameter and keep only required one
%change the parameter to englobe the full arrear.
%params.oversize_z
output_size_pixel=params.size(:);
creation_size_pixel=output_size_pixel;
creation_size_pixel=creation_size_pixel.*[sqrt(2)*1.1 sqrt(2)*1.1 1]' + params.oversize_z;
creation_size=creation_size_pixel(:).*params.resolution(:);
max_size=max(creation_size(1:2));
creation_size_pixel(1:2)=ceil(max_size./params.resolution(1:2));
params.size=creation_size_pixel;
%create the utility variable
warning('off','all');
utility=DERIVE_OPTICAL_TOOL(params,params.use_GPU);
warning('on','all');
%compute green

range=1;
samples=params.oversample_xy.*params.oversample_xy_fourier;%round(params.oversample_xy.*min(params.size(1)/2,params.size(2)/2));
if samples<(sqrt(params.size(1).^2+params.size(2).^2)*5)
    error('the code might not be valid for such big array maybee increase the : params_required.oversample_xy')
end
%error('use two truncated small hankel matrix for speed and memory instead of 1 big');
hankler=discrete_hankel_croped_fourier(range,samples,0,(sqrt(params.size(1).^2+params.size(2).^2).*params.oversample_xy_fourier)/(2*samples),params.use_GPU);
%hankler=discrete_hankel(range,samples,0,true);

H_optical_coor=hankler.get_r();
H_optical_coor=H_optical_coor./max(H_optical_coor(:));
H_optical_coor=H_optical_coor.*max(utility.fourier_space.coorxy(:)).*params.oversample_xy_fourier;
H_optical_k3=(utility.k0_nm).^2-(H_optical_coor).^2;H_optical_k3(H_optical_k3<0)=0;H_optical_k3=sqrt(H_optical_k3);
H_optical_circle=H_optical_coor<utility.kmax;

green=sinc((H_optical_k3-utility.fourier_space.coor{3})./utility.fourier_space.res{3}...
    .*((utility.fourier_space.size{3}-params.oversize_z/2)/utility.fourier_space.size{3}))...
    .*((utility.fourier_space.size{3}-params.oversize_z/2)/utility.fourier_space.size{3});
if ~refocus_version
    green=green.*H_optical_circle./(H_optical_k3+~H_optical_circle);
    green=green./(1i*4*pi);
else
    green=green.*H_optical_circle;
end
green=squeeze(green);

img_green=hankler.forward(green);
normalised_k=(hankler.get_k()./max(hankler.get_k(),[],'all'));%.*(samples/(sqrt(params.size(1).^2+params.size(2).^2)/2));

%mask=(normalised_k<1);
mask=(normalised_k<0.9)+(normalised_k<1).*(normalised_k>0.9).*(1-normalised_k).*10;
%figure; plot(mask);

img_green=img_green.*mask;
green=hankler.backward(img_green);

clear hankler;
%figure; imagesc(abs(green));
[nearest_coordinate,nearest_coordinate2,blending_coeff]=bin_search_nearest(utility.fourier_space.coorxy,H_optical_coor);
%nearest_coordinate=nearest_coordinate+...
%    reshape(length(H_optical_coor).*((1:(utility.fourier_space.size{3}))-1),1,1,[]);
%nearest_coordinate2=nearest_coordinate2+...
%    reshape(length(H_optical_coor).*((1:(utility.fourier_space.size{3}))-1),1,1,[]);

%figure; imagesc(blending_coeff);axis image;
green_out=zeros(output_size_pixel(1),output_size_pixel(2),utility.fourier_space.size{3},'single');
if params.use_GPU
    green_out=gpuArray(green_out);
end

slices=1:params.simultanous_process:utility.fourier_space.size{3};
if length(slices)==1 || slices(end)~=utility.fourier_space.size{3}
    slices(end+1)=utility.fourier_space.size{3};
end
for ii=1:length(slices)-1
    z_index=slices(ii):slices(ii+1);
    z_coor_add=reshape(length(H_optical_coor).*((z_index)-1),1,1,[]);
    green_slice=green(nearest_coordinate+z_coor_add).*blending_coeff+green(nearest_coordinate2+z_coor_add).*(1-blending_coeff);
    green_slice=green_slice.*utility.fourier_space.size{3}./(utility.image_space.res{1}.*utility.image_space.res{2});
     
    %cut the green function
    green_slice=fftshift(ifft2(ifftshift(green_slice)));
    ROI_start=(floor([size(green_slice,1) size(green_slice,2)]/2)+1)-(floor(output_size_pixel(1:2)/2));
    ROI_end=ROI_start+output_size_pixel(1:2)-1;
    green_slice=green_slice(...
        ROI_start(1):ROI_end(1),...
        ROI_start(2):ROI_end(2),...
        :);
    green_slice=fftshift(fft2(ifftshift(green_slice)));
    green_out(:,:,z_index)=green_slice;
end
green_out=fftshift(ifft(ifftshift(green_out),size(green_out,3),3));

ROI_start=(floor(size(green_out)'/2)+1)-(floor(output_size_pixel(:)/2));
ROI_end=ROI_start+output_size_pixel(:)-1;
green_out=green_out(:,:,ROI_start(3):ROI_end(3));

green_out=fftshift(fft(ifftshift(green_out),size(green_out,3),3));
end
function [nearest_coordinate,nearest_coordinate2,blending_coeff]=bin_search_nearest(value,searched_pos)
searched_pos=searched_pos(:);
nearest_coordinate=0.*value+1;
dist=length(searched_pos);
step_up=dist;
while step_up>1
    step_up=ceil(step_up/2);
    upper_coordinate=nearest_coordinate+step_up;
    upper_coordinate(upper_coordinate>dist)=dist;
    change_flag=(searched_pos(upper_coordinate)<value);
    nearest_coordinate=upper_coordinate.*change_flag+nearest_coordinate.*~change_flag;
end
nearest_coordinate2=nearest_coordinate;
nearest_coordinate2(searched_pos(nearest_coordinate)<value)=nearest_coordinate2(searched_pos(nearest_coordinate)<value)+1;
nearest_coordinate2(nearest_coordinate2>dist)=dist;

blending_coeff=(value-searched_pos(nearest_coordinate))./(searched_pos(nearest_coordinate2)-searched_pos(nearest_coordinate));
blending_coeff(nearest_coordinate2==nearest_coordinate)=1;
end

