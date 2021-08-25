classdef FORWARD_SOLVER < handle
    properties (SetAccess = protected, Hidden = true)
        parameters;
        RI;
        
        utility;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            %SIMULATION PARAMETERS
            params.return_transmission=true;%return transmission field
            params.return_reflection=false;%return reflection field
            params.return_3D=false;%return 3D field
            params.use_GPU=true;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER(params)
            h.parameters=params;
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters,h.parameters.use_GPU);%reset it to have the gpu used
            warning('on','all');
        end
        function set_RI(h,RI)
            h.RI=RI;
        end
        function [field_trans,field_ref,field_3D]=solve(h,input_field)
            error("You need to specify the forward solver to solve");
            field_trans=[];
            field_ref=[];
            field_3D=[];
        end
        function fft_Field_3pol=transform_field_3D(h,fft_Field_2pol)
            ZP=size(fft_Field_2pol);
            fft_Field_2pol=fft_Field_2pol.*h.utility.NA_circle;
            
            if h.parameters.use_abbe_sine
                %abbe sine condition is due to the magnification
                
                filter=single(h.utility.NA_circle);
                filter(h.utility.NA_circle)=filter(h.utility.NA_circle)./sqrt(h.utility.cos_theta(h.utility.NA_circle));
                fft_Field_2pol=fft_Field_2pol.*filter;
            end
            if size(fft_Field_2pol,3)>1
                if ZP(3)~=2
                    error('Far field has two polarisation');
                end
                
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,h.parameters.RI_bg,h.parameters.wavelength,h.parameters.resolution);
                
                fft_Field_2pol=fft_Field_2pol.*K_mask;
                
                Field_new_basis=zeros(ZP(1),ZP(2),2,size(fft_Field_2pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_2pol.*Radial_2D,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_2pol.*Perp_2D,3);
                
                fft_Field_3pol=zeros(ZP(1),ZP(2),3,size(fft_Field_2pol,4),'single');%the field in the 3D
                fft_Field_3pol         =fft_Field_3pol          + Field_new_basis(:,:,1,:).*ewald_TanVec;
                fft_Field_3pol(:,:,1:2,:)=fft_Field_3pol(:,:,1:2,:) + Field_new_basis(:,:,2,:).*Perp_2D;
            else
                fft_Field_3pol=fft_Field_2pol;
            end
        end
        function fft_Field_2pol=transform_field_2D(h,fft_Field_3pol)
            ZP=size(fft_Field_3pol);
            fft_Field_3pol=fft_Field_3pol.*h.utility.NA_circle;
            
            if h.parameters.use_abbe_sine
                %abbe sine condition is due to the magnification
                
                fft_Field_3pol=fft_Field_3pol.*sqrt(h.utility.cos_theta).*h.utility.NA_circle;
            end
            if size(fft_Field_3pol,3)>1
                if ZP(3)~=3
                    error('Near field has three polarisation');
                end
                
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,h.parameters.RI_bg,h.parameters.wavelength,h.parameters.resolution);
                
                fft_Field_3pol=fft_Field_3pol.*K_mask;
                
                Field_new_basis=zeros(ZP(1),ZP(2),2,size(fft_Field_3pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_3pol         .*ewald_TanVec,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_3pol(:,:,1:2,:).*Perp_2D,3);
                
                fft_Field_2pol=zeros(ZP(1),ZP(2),2,size(fft_Field_3pol,4),'single');%the field in the 2D
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,1,:).*Radial_2D;
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,2,:).*Perp_2D;
            else
                fft_Field_2pol=fft_Field_3pol;
            end
        end
        function fft_Field_2pol=transform_field_2D_reflection(h,fft_Field_3pol)
            ZP=size(fft_Field_3pol);
            fft_Field_3pol=fft_Field_3pol.*h.utility.NA_circle;
            
            if h.parameters.use_abbe_sine
                %abbe sine condition is due to the magnification
                
                fft_Field_3pol=fft_Field_3pol.*sqrt(h.utility.cos_theta).*h.utility.NA_circle;
            end
            if size(fft_Field_3pol,3)>1
                if ZP(3)~=3
                    error('Near field has three polarisation');
                end
                
                [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,h.parameters.RI_bg,h.parameters.wavelength,h.parameters.resolution);
                
                ewald_TanVec(:,:,3)=-ewald_TanVec(:,:,3);%because reflection invers k3
                
                fft_Field_3pol=fft_Field_3pol.*K_mask;
                
                Field_new_basis=zeros(ZP(1),ZP(2),2,size(fft_Field_3pol,4),'single');%the field in the polar basis
                Field_new_basis(:,:,1,:)=sum(fft_Field_3pol         .*ewald_TanVec,3);
                Field_new_basis(:,:,2,:)=sum(fft_Field_3pol(:,:,1:2,:).*Perp_2D,3);
                
                fft_Field_2pol=zeros(ZP(1),ZP(2),2,size(fft_Field_3pol,4),'single');%the field in the 2D
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,1,:).*Radial_2D;
                fft_Field_2pol=fft_Field_2pol+Field_new_basis(:,:,2,:).*Perp_2D;
            else
                fft_Field_2pol=fft_Field_3pol;
            end
        end
    end
end
function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = polarisation_utility(ZP,n_m,lambda,dx)

ZP=ZP(1:2);

k0=1/lambda; %[um-1, spatial frequency @ vacuum]
k =2*pi*n_m*k0; % [um-1, spatial wavenumber @ medium ]

kres=1./dx(1:2)./ZP(1:2);

K_1=single(2*pi*kres(1)/k*(-floor(ZP(1)/2):ZP(1)-floor(ZP(1)/2)-1));%normalised to diffraction limit k1
K_2=single(2*pi*kres(2)/k*(-floor(ZP(2)/2):ZP(1)-floor(ZP(2)/2)-1));%normalised to diffraction limit k2

K_1=reshape(K_1,[],1);
K_2=reshape(K_2,1,[]);

K_3=real(sqrt(1-(K_1.^2+K_2.^2)));
K_mask = ~(K_3==0);

Radial_2D=zeros(ZP(1),ZP(2),2,'single');
Perp_2D=zeros(ZP(1),ZP(2),2,'single');
norm_rad=sqrt(K_1.^2+K_2.^2);

temp1=K_1./norm_rad;
temp2=K_2./norm_rad;
temp1(norm_rad==0)=1;
temp2(norm_rad==0)=0;

Radial_2D(:,:,1)=temp1;
Radial_2D(:,:,2)=temp2;
clear temp1;
clear temp2;

Perp_2D(:,:,1)=Radial_2D(:,:,2);
Perp_2D(:,:,2)=-Radial_2D(:,:,1);

Radial_3D=zeros(ZP(1),ZP(2),3,'single');
norm_sph=sqrt(K_1.^2+K_2.^2+K_3.^2);
Radial_3D(:,:,1)=K_1./norm_sph;
Radial_3D(:,:,2)=K_2./norm_sph;
Radial_3D(:,:,3)=K_3./norm_sph;

ewald_TanProj=sum(Radial_3D(:,:,1:2).*Radial_2D,3);
ewald_TanVec=zeros(ZP(1),ZP(2),3,'single');

ewald_TanVec(:,:,1:2)=Radial_2D(:,:,:);
ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
ewald_TanVec_norm(~K_mask)=1;
ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;

end