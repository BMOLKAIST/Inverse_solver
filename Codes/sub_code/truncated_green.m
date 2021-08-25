function green=truncated_green(params)
%inspired from : "Fast convolution with free-space Green’s functions" Felipe Vico Leslie Greengard Miguel Ferrando
%check that input has all the requires params
params_required=BASIC_OPTICAL_PARAMETER();
params_required.use_GPU=true;
%params_required
%params
params=update_struct(params_required,params);%check for reuired parameter and keep only required one
%create the utility variable
warning('off','all');
utility=DERIVE_OPTICAL_TOOL(params,params.use_GPU); 
warning('on','all');
%compute green
warning('handle even and odd');
warning('treat the s==k case');
warning('cut from round to square');
if mod(utility.fourier_space.size{1},2)~=1
    error('not implemented for even size yet');
end
S=2.*pi.*sqrt(abs(utility.fourier_space.coor{1}).^2+abs(utility.fourier_space.coor{2}).^2+abs(utility.fourier_space.coor{3}).^2);
%{
L=utility.image_space.size{1}.*utility.image_space.res{1}./3;
N3=utility.image_space.size{1}.*utility.image_space.size{2}.*utility.image_space.size{3};
K=2.*pi.*utility.k0_nm;
green=(1-...
    exp(1i.*K.*L)...
    .*(cos(S.*L)-1i.*K.*L.*sinc(S.*L))...
    )...
    ./((S-K).*(S+K))...
    .*sqrt(N3);  
%}
L=norm([utility.image_space.size{1}.*utility.image_space.res{1} utility.image_space.size{2}.*utility.image_space.res{2} utility.image_space.size{3}.*utility.image_space.res{3}]);
L = L/5;
N3=utility.image_space.size{1}.*utility.image_space.size{2}.*utility.image_space.size{3};
K=2.*pi.*utility.k0_nm;
% green=(1-...
%     exp(1i.*K.*L)...
%     .*(cos(S.*L)-1i.*K.*L.*sinc(S.*L/pi))...
%     )...
%     ./(S.^2-K.^2) * sqrt(N3);  
green=(1 ./(S.^2-K.^2)...
    -1./2./S...
    .*(exp(1i.*(S+K).*L)./(S+K)+exp(1i.*(-S+K).*L)./(S-K))...
    )* sqrt(N3);  
green(S==0) = -(1-exp(1i*K*L)+1i*K*L*exp(1i*K*L)) / K^2;


% %{
green(S==K)= 1/2/K.*...
    (1i*L + ...
    (1-exp(1i.*K.*L)) ./ 2 ./ K...
    );
    %}
%green=abs(coor_radius-utility.k0);

end
