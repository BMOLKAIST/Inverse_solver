classdef FORWARD_SOLVER_BPM < FORWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        %utility;
        field_multiplier;
        refocus_kernel;
        filter_out;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@FORWARD_SOLVER();
            %specific parameters
            params.iterations_number=-1;
            params.use_GPU=true;
            params.cos_correction=true;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER_BPM(params)
            h@FORWARD_SOLVER(params);
        end
        function set_RI(h,RI)
            RI=single(RI);%single computation are faster
            set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            if ~h.parameters.use_GPU
                h.RI=gpuArray(h.RI);
            end
            %warning('off','all');
            %h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            %warning('on','all');
            
            h.field_multiplier=(1i.*2.*pi.*h.utility.k0.*(h.RI-h.parameters.RI_bg).*h.parameters.resolution(3));
            h.filter_out=ifftshift(h.utility.NA_circle);
            h.refocus_kernel=ifftshift(exp(h.utility.refocusing_kernel.*h.parameters.resolution(3)).*h.utility.NA_circle);
            %h.filter_out=ifftshift(imag(h.utility.refocusing_kernel)>0);
            %h.refocus_kernel=ifftshift(exp(h.utility.refocusing_kernel.*h.parameters.resolution(3)).*(imag(h.utility.refocusing_kernel)>0));
            if ~h.parameters.use_GPU
                h.field_multiplier=gpuArray(single(h.field_multiplier));
                h.refocus_kernel=gpuArray(single(h.refocus_kernel));
            end
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            if ~h.parameters.use_GPU
                h.parameters.use_cuda=false;
                input_field=single(input_field);
            else
                input_field=single(gpuArray(input_field));
            end
            %find the max pos
            if h.parameters.cos_correction
                [~,Indexes_angle] = max(abs(fftshift(fft2(ifftshift(input_field)))),[],[1 2 3],'linear');
                Indexes_angle=1+mod(Indexes_angle-1,size(input_field,1).*size(input_field,2));
                cos_coeffs=h.utility.cos_theta(Indexes_angle);
                cos_coeffs=reshape(cos_coeffs,1,1,1,[]);
                if ~h.parameters.use_GPU
                    cos_coeffs=gpuArray(cos_coeffs);
                end
            else
                
            end
            %solve
            out_pol=1;
            out_pol_3D=1;
            if size(input_field,3)>1
                if size(input_field,3)~=2
                    error('must give far field polarisation with 2 pols only');
                end
                error('Look at the old version for polarised version but it is useless')
                
            end
            
            fields_trans=[];
            if h.parameters.return_transmission
                fields_trans=ones(size(h.field_multiplier,1),size(h.field_multiplier,2),out_pol,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.parameters.return_reflection
                fields_ref=ones(size(h.field_multiplier,1),size(h.field_multiplier,2),out_pol,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.parameters.return_3D
                fields_3D=ones(size(h.field_multiplier,1),size(h.field_multiplier,2),size(h.field_multiplier,3),out_pol_3D,size(input_field,4),'single');
            end
            
            %refocus and go to 3D field
            input_field=fftshift(fft2(ifftshift(input_field)));
            input_field=input_field.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(-floor(size(h.field_multiplier,3)/2)-1));
            input_field=fftshift(ifft2(ifftshift(input_field)));
            %compute
            %refocus
            input_field=fft2(input_field);
            for step=1:size(h.field_multiplier,3)
                
                input_field=input_field.*h.refocus_kernel;
                %figure; imagesc(abs(h.refocus_kernel));figure; imagesc(angle(h.refocus_kernel));error('stop');
                input_field=ifft2(input_field);
                %apply phase
                if h.parameters.cos_correction
                    input_field=input_field.*exp((1./cos_coeffs).*h.field_multiplier(:,:,step));
                else
                    input_field=input_field.*exp(h.field_multiplier(:,:,step));
                end
                %refocus
                input_field=fft2(input_field);
                %save volume
                if h.parameters.return_3D
                    fields_3D(:,:,step,:,:)=gather(reshape(ifft2(input_field.*h.filter_out),size(input_field,1),size(input_field,2),1,size(input_field,3),size(input_field,4)));
                end
            end
            input_field=ifft2(input_field);
            %refocus and go 2D
            input_field=fftshift(fft2(ifftshift(input_field)));
            input_field=input_field.*exp(h.utility.refocusing_kernel.*h.parameters.resolution(3).*(-(size(h.field_multiplier,3)-floor(size(h.field_multiplier,3)/2)-1)));
            input_field=fftshift(ifft2(ifftshift(input_field)));
            fields_trans=gather(input_field);
        end
    end
end
