classdef BACKWARD_SOLVER_RYTOV_multifocus < BACKWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        utility;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@BACKWARD_SOLVER();
            %specific parameters
            
            params.use_non_negativity=false;
            params.non_negativity_iteration=100;
            params.Zstep_size = 1000;
            
            
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=BACKWARD_SOLVER_RYTOV_multifocus(params)
            h@BACKWARD_SOLVER(params);
        end
        function [RI, ORytov]=solve(h,input_field,output_field)
            Zstep_size = h.parameters.Zstep_size;
%             if nargin == 3
%                 Zstep_size = 1000;
%             end
            
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            
            if length(size(input_field))~=4
               error('You need to provide the field with 4 dimenssion : dim1 x dim2 x polarisation x illuminationnumber') 
            end
            if size(input_field,1)~=size(input_field,2)
                error('Please input a square field');
            end
            if ~isequal(size(input_field),size(output_field))
                error('Please input field and bg of same size');
            end
            if h.parameters.resolution(1)~=h.parameters.resolution(2)
                error('x/y input resolution must be isotropic');
            end
            if h.parameters.size(1)~=h.parameters.size(2)
                error('x/y output size must be isotropic');
            end
            if h.parameters.size(1)~=size(input_field,1) || h.parameters.size(2)~=size(input_field,2)
                error('declare size in the parameter must be the same as the field size');
            end

                        
            [bg,sp]=vector2scalarfield(input_field,output_field);
            field = sp ./ bg;
            thetaSize=size(field,3);
            num_z = 1 + max(0,floor((h.parameters.size(3)-1-Zstep_size)/Zstep_size/2))*2;
            Zs = ((1:num_z) - floor(num_z/2)-1);
            Emask = (h.utility.fourier_space.coorxy)<(2*h.parameters.NA/h.parameters.wavelength);
            Zmask_total = zeros(1,1,h.parameters.size(3),'single');
            Zmask=(1:h.parameters.size(3))-(floor(h.parameters.size(3)/2)+1);
            Zmask = reshape(Zmask, [1 1 length(Zmask)]);
            Zmask = Zstep_size/2-abs(Zmask);
            Zmask(Zmask<=0) = 1;
            
            if h.parameters.use_GPU
                RI=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single','gpuArray')));
                ORytov=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single','gpuArray')));
                Count=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single','gpuArray')));
            else
                RI=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single')));
                ORytov=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single')));
                Count=(single(zeros(h.parameters.size(1),h.parameters.size(2),h.parameters.size(3),'single')));
            end
            M_z = min(Zs);
            disp(['Number of multifocus is: ' num2str(num_z)])
            % find angle
            f_dx=zeros(thetaSize,1);
            f_dy=zeros(thetaSize,1);
            mj_0=0;
            mi_0=0;
            for ii=1:size(bg,3)
                Fbg=fftshift(fft2(ifftshift(bg(:,:,ii))));
                [mj,mi]=find(Fbg==max(Fbg(:)));
                if ii==1
                    mi_0=mi;
                    mj_0=mj;
                end
                f_dy(ii)=mj-mj_0;
                f_dx(ii)=mi-mi_0;
            end
            k0_x=h.utility.fourier_space.res{2}.*f_dx;
            k0_y=h.utility.fourier_space.res{1}.*f_dy;
            k0_z=real(sqrt((h.utility.k0_nm)^2-(k0_x).^2-(k0_y).^2));
            
            for j1 = 1:num_z
                if num_z > 1
                    (M_z+j1-1)*Zstep_size
                    sp = h.refocus(output_field, (M_z+j1-1)*Zstep_size.*h.parameters.resolution(3));
                    bg = h.refocus(input_field, (M_z+j1-1)*Zstep_size.*h.parameters.resolution(3));
                    [bg,sp]=vector2scalarfield(bg,sp);
                    field = sp ./ bg;
                end
                
                retAmplitude=abs(field);
                retPhase=angle(field);
                retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
                if h.parameters.PhiShift_on
                    retPhase = PhiShiftMS(retPhase,1,1);
                end
                
                
                for kk= 1 :thetaSize
                    FRytov=squeeze(log(retAmplitude(:,:,kk))+1i*retPhase(:,:,kk));
                    UsRytov=fftshift(fft2(ifftshift(FRytov))).*(h.parameters.resolution(1).*h.parameters.resolution(2)); % unit: (um^2)
                    UsRytov=gpuArray(circshift(UsRytov,[round(k0_y(kk)/h.utility.fourier_space.res{1}) round(k0_x(kk)/h.utility.fourier_space.res{2})]));
                    UsRytov=UsRytov.*h.utility.NA_circle;
                    size_check=zeros(h.parameters.size(1:2),'single');
                    kz=h.utility.k3+size_check;
                    kx=h.utility.fourier_space.coor{2}+size_check;
                    ky=h.utility.fourier_space.coor{1}+size_check;
                    Kx=kx-k0_x(kk);Ky=ky-k0_y(kk);Kz=kz-k0_z(kk);
                    Uprime= (kz/1i).*UsRytov; % unit: (um^1) % kz is spatial frequency, so 2pi is multiplied for wave vector
                    xind=find((kz>0).*h.utility.NA_circle...
                        .*(Kx>(h.utility.fourier_space.res{2}*(-floor(h.parameters.size(2)/2))))...
                        .*(Ky>(h.utility.fourier_space.res{1}*(-floor(h.parameters.size(1)/2))))...
                        .*(Kz>(h.utility.fourier_space.res{3}*(-floor(h.parameters.size(3)/2))))...
                        .*(Kx<(h.utility.fourier_space.res{2}*(floor(h.parameters.size(2)/2)-1)))...
                        .*(Ky<(h.utility.fourier_space.res{1}*(floor(h.parameters.size(1)/2)-1)))...
                        .*(Kz<(h.utility.fourier_space.res{3}*(floor(h.parameters.size(3)/2)-1))));
                    Uprime=Uprime(xind); Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind);
                    Kx=round(Kx/h.utility.fourier_space.res{2}+floor(h.parameters.size(2)/2)+1); 
                    Ky=round(Ky/h.utility.fourier_space.res{1}+floor(h.parameters.size(1)/2)+1); 
                    Kz=round(Kz/h.utility.fourier_space.res{3}+floor(h.parameters.size(3)/2)+1);
                    Kzp=sub2ind(size(Count),Ky,Kx,Kz);
                    temp=ORytov(Kzp);
                    ORytov(Kzp)=temp+Uprime;
                    Count(Kzp)=Count(Kzp)+(Uprime~=0);
                    %disp([num2str(kk),' / ',num2str(thetaSize)])
                end
                ORytov(Count>0)=ORytov(Count>0)./Count(Count>0)/(prod(h.parameters.resolution(:),'all')); % should be (um^-2)*(px*py*pz), so (px*py*pz/um^3) should be multiplied.
                Reconimg=(fftshift(ifftn(ifftshift(ORytov))));
                Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
                Count(:) = 0;

                if h.parameters.use_non_negativity
                    Reconimg = (fftshift(ifftn(ifftshift(ORytov))));
                    for mm = 1 : h.parameters.non_negativity_iteration
                        Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                        ORytov_new=fftshift(fftn(ifftshift(Reconimg)));
                        ORytov_new=Emask.*ORytov_new.*(abs(ORytov)==0)+ORytov;
                        Reconimg=fftshift(ifftn(ifftshift(ORytov_new)));
                        %disp([num2str(mm),' / ',num2str(h.parameters.non_negativity_iteration)])
                    end
                    Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                    Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
                end
%                 M_z
%                 orthosliceViewer(gather(real(Reconimg))),pause
                
                Zmask_total = Zmask_total + single(circshift(Zmask, [0 0 (M_z+j1-1)*Zstep_size]));
%                 Reconimg = circshift(Reconimg, [0 0 M_z*Zstep_size]);
                RI = RI + (circshift(Reconimg.*Zmask, [0 0 (M_z+j1-1)*Zstep_size]));
%                 RI = RI + (circshift(Reconimg.*Zmask, [0 0 M_z*Zstep_size]));
                ORytov(:) = 0;
            end
            
            RI=gather(RI./Zmask_total);
            if num_z > 1
                ORytov_new = ORytov;
                ORytov = RI2potential(Reconimg,h.parameters.wavelength,h.parameters.RI_bg);
                ORytov = fftshift(fftn(ifftshift(ORytov))).*(ORytov_new~=0);
                if j1 <num_z
                    ORytov(:) = 0;
                end
            else
                ORytov = gather(ORytov);
            end
        end

        function [field_trans_f] = refocus(h, field_trans, z)
            field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(field_trans))) .* exp(z.*h.utility.refocusing_kernel))));
        end
        
    end
end