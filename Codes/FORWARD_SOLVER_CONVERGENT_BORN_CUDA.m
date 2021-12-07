classdef FORWARD_SOLVER_CONVERGENT_BORN_CUDA < FORWARD_SOLVER_CONVERGENT_BORN
    properties %(SetAccess = protected, Hidden = true)
       cuda_id;
       used_gpu;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@FORWARD_SOLVER_CONVERGENT_BORN();
            %specific parameters
            params.simultanous_2D_fft=10;%the number of 2D fft to do at one time (to save memory do small batch)
            params.used_gpu=0;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER_CONVERGENT_BORN_CUDA(params)
            
            params.use_GPU=false; %the gpu part is done in cuda 
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h@FORWARD_SOLVER_CONVERGENT_BORN(params);
            cuda_params=struct;
            cuda_params.simulation_size=h.expected_RI_size(:)+2*h.boundary_thickness_pixel(:);
            cuda_params.RI_size=h.expected_RI_size(:);
            cuda_params.field_size=params.size(:);
            cuda_params.field_size(3)=1;
            cuda_params.conv_size=cuda_params.RI_size+cuda_params.field_size;
            cuda_params.conv_size(3)=params.simultanous_2D_fft;
            cuda_params.green_size=floor(cuda_params.conv_size/2)+1;
            cuda_params.green_size(3)=floor(cuda_params.RI_size(3)/2)+1;
            cuda_params.used_gpu=uint32(params.used_gpu);
            h.used_gpu=params.used_gpu;
            green_for_cuda=h.refocusing_util(1:cuda_params.green_size(1),1:cuda_params.green_size(2),1:cuda_params.green_size(3));
            green_for_cuda_2=h.kernel_trans(1:cuda_params.green_size(1),1:cuda_params.green_size(2),1:cuda_params.green_size(3));
            
            %a=gpuDevice();
            h.cuda_id=CudaBorn(uint32(0),uint32(0),cuda_params,green_for_cuda,green_for_cuda_2);
            %gpuDevice(a.Index);
            
        end
        function set_RI(h,RI)
            if ~isequal(size(RI)',h.expected_RI_size(:))
                error(['The refractiv index does not have the expected size : ' ...
                    num2str(h.expected_RI_size(1)) ' ' num2str(h.expected_RI_size(2)) ' ' num2str(h.expected_RI_size(3))]);
            end
            RI=single((RI));%single computation are faster
            set_RI@FORWARD_SOLVER_CONVERGENT_BORN(h,RI);%call the parent class function to save the RI
            
            cuda_params=struct;
            cuda_params.k0_nm=single(h.utility_border.k0_nm);
            cuda_params.eps_imag=single(h.eps_imag);
            cuda_params.fourier_res1=single(h.utility_border.fourier_space.res{1});
            cuda_params.fourier_res2=single(h.utility_border.fourier_space.res{2});
            cuda_params.fourier_res3=single(h.utility_border.fourier_space.res{3});
            cuda_params.ROI=h.ROI;
            cuda_params.Bornmax=(h.Bornmax);
            %cuda_params.Bornmax
            cuda_params.dV=h.utility_border.dV;
            %a=gpuDevice();
            CudaBorn(uint32(1),uint32(h.cuda_id),cuda_params,single(h.V));
            %gpuDevice(a.Index);
        end
        function delete(h)
            %a=gpuDevice();
            CudaBorn(uint32(3),uint32(h.cuda_id));
            %gpuDevice(a.Index);
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            if ~h.parameters.use_GPU
                input_field=single(input_field);
            else
                h.RI=single(gpuArray(h.RI));
                input_field=single(gpuArray(input_field));
            end
            if size(input_field,3)>1 &&~h.parameters.vector_simulation
                error('the source is 2D but parameter indicate a vectorial simulation');
            elseif size(input_field,3)==1 && h.parameters.vector_simulation
                error('the source is 3D but parameter indicate a non-vectorial simulation');
            end
            if h.parameters.verbose && size(input_field,3)==1
                warning('Input is scalar but scalar equation is less precise');
            end
            if size(input_field,3)>2
                error('Input field must be either a scalar or a 2D vector');
            end
            
            input_field=fftshift(fft2(ifftshift(input_field)));
            %2D to 3D field
            [input_field] = h.transform_field_3D(input_field);
            %compute
            out_pol=1;
            if h.pole_num==3
                out_pol=2;
            end
            fields_trans=[];
            if h.parameters.return_transmission
                fields_trans=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.parameters.return_reflection
                fields_ref=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),out_pol,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.parameters.return_3D
                fields_3D=ones(1+h.ROI(2)-h.ROI(1),1+h.ROI(4)-h.ROI(3),1+h.ROI(6)-h.ROI(5),size(input_field,3),size(input_field,4),'single');
            end
            num_gpu=length(h.used_gpu);
            if num_gpu==0
                num_gpu=1;
            end
            num_gpu
            for itt_num=1:ceil(size(input_field,4)/num_gpu)
                start_field=((itt_num-1)*length(h.used_gpu))+1;
                end_field=(itt_num)*length(h.used_gpu);
                end_field=min(end_field,size(input_field,4));
                field_num=start_field:end_field;
                
                loaded_gpu=(0:length(field_num)-1);
                
                %a=gpuDevice();
                [field_trans,field_ref,Field]=CudaBorn(uint32(4),uint32(h.cuda_id),complex(fftshift(ifft2(ifftshift(input_field(:,:,:,field_num))))),uint32(loaded_gpu));
                %gpuDevice(a.Index);
                
                field_trans=reshape(field_trans,size(field_trans,1),size(field_trans,2),size(input_field,3),[]);
                field_ref=reshape(field_ref,size(field_ref,1),size(field_ref,2),size(input_field,3),[]);
               
                Field=reshape(Field,size(Field,1),size(Field,2),size(Field,3),size(input_field,3),[]);
                if h.parameters.return_3D
                    fields_3D(:,:,:,:,field_num)=(Field);
                end
                if h.parameters.return_transmission
                    field_trans=fftshift(fft2(ifftshift(field_trans)));
                    field_trans=field_trans+input_field(:,:,:,field_num);
                    [field_trans] = h.transform_field_2D(field_trans);
                    field_trans=fftshift(ifft2(ifftshift(field_trans)));
                    fields_trans(:,:,:,field_num)=field_trans;
                    %fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
                end
                if h.parameters.return_reflection
                    field_ref=fftshift(fft2(ifftshift(field_ref)));
                    [field_ref] = h.transform_field_2D_reflection(field_ref);
                    field_ref=fftshift(ifft2(ifftshift(field_ref)));
                    fields_ref(:,:,:,field_num)=field_ref;
                    %fields_ref(:,:,:,field_num)=gather(squeeze(field_ref));
                end
            end
        end
    end
end


