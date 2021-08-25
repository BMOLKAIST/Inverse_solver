function green=truncated_green(params)
%inspired from : "Fast convolution with free-space Green’s functions" Felipe Vico Leslie Greengard Miguel Ferrando
%check that input has all the requires params
params_required=BASIC_OPTICAL_PARAMETER();
params_required.use_GPU=true;
%params_required
%params
params=update_struct(params_required,params);%check for reuired parameter and keep only required one
%change the parameter to englobe the full arrear.
output_size_pixel=params.size(:);
output_size=output_size_pixel(:).*params.resolution(:);
max_size=max(output_size(:));
creation_size_pixel=ceil(max_size.*(1+(sqrt(3)-1)/2)./params.resolution(:))+4;%+2 to remove boundary effect to to the center beeing a bit of center for even/odd size matts
params.size=creation_size_pixel;
%create the utility variable
warning('off','all');
utility=DERIVE_OPTICAL_TOOL(params,params.use_GPU); 
warning('on','all');
%compute green
S=2.*pi.*sqrt(abs(utility.fourier_space.coor{1}).^2+abs(utility.fourier_space.coor{2}).^2+abs(utility.fourier_space.coor{3}).^2);
L=norm([utility.image_space.size{1}.*utility.image_space.res{1} utility.image_space.size{2}.*utility.image_space.res{2} utility.image_space.size{3}.*utility.image_space.res{3}]);
L=L/sqrt(3);
L=L/2;
L=L.*sqrt(3)./(1+(sqrt(3)-1)/2);
K=2.*pi.*utility.k0_nm; 
green=(1 ./(S.^2-K.^2)...
    -1./2./S...
    .*(exp(1i.*(S+K).*L)./(S+K)+exp(1i.*(-S+K).*L)./(S-K))...
    );
green(S==0) = -(1-exp(1i*K*L)+1i*K*L*exp(1i*K*L)) / K^2;
green(S==K)= 1/2/K.*...
    (1i*L + ...
    (1-exp(1i.*K.*L)) ./ 2 ./ K...
    );
green=green.*utility.dVk;
%cut the green function
green=fftshift(ifftn(ifftshift(green)));
ROI_start=(floor(size(green)'/2)+1)-(floor(output_size_pixel(:)/2));
ROI_end=ROI_start+output_size_pixel(:)-1;
green=green(...
    ROI_start(1):ROI_end(1),...
    ROI_start(2):ROI_end(2),...
    ROI_start(3):ROI_end(3));
green=fftshift(fftn(ifftshift(green)));
end
