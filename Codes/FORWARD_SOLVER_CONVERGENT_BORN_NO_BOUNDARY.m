classdef FORWARD_SOLVER_CONVERGENT_BORN_NO_BOUNDARY < FORWARD_SOLVER
    properties %(SetAccess = protected, Hidden = true)
        utility_border;
        Bornmax;
        boundary_thickness_pixel;
        ROI;
        
        Greenp;
        rads;
        eye_3;
        
        V;
        pole_num;
        green_absorbtion_correction;
        eps_imag;
        
        kernel_trans;
        kernel_ref;
        
        attenuation_mask;
        pixel_step_size;
        
        phase_ramp;
        
        refocusing_util;
        
        expected_RI_size;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@FORWARD_SOLVER();
            %specific parameters
            params.iterations_number=-1;
            params.boundary_thickness = 6.*[1 1 1];
            params.boundary_sharpness = 1;%2;
            params.verbose = false;
            params.acyclic = true;
            params.RI_xy_size=[0 0];%if set to 0 the field is the same size as the simulation
            params.RI_center=[0 0];
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER_CONVERGENT_BORN_NO_BOUNDARY(params)
            
            % make the refocusing to volume field(other variable depend on the max RI and as such are created later).
            h@FORWARD_SOLVER(params);
            if h.parameters.RI_xy_size(1)==0
                h.parameters.RI_xy_size(1)=h.parameters.size(1);
            end
            if h.parameters.RI_xy_size(2)==0
                h.parameters.RI_xy_size(2)=h.parameters.size(2);
            end
            h.expected_RI_size=[h.parameters.RI_xy_size(1) h.parameters.RI_xy_size(2) h.parameters.size(3)];
            
            h.refocusing_util=exp(h.utility.refocusing_kernel.*h.utility.image_space.coor{3});
            %h.refocusing_util=truncated_z_refocusing(h.parameters);h.refocusing_util=fftshift(ifft(ifftshift(h.refocusing_util),size(h.refocusing_util,3),3));
            h.refocusing_util=gather(h.refocusing_util);
            %figure;orthosliceViewer(squeeze(abs(h.refocusing_util)));
            %figure;orthosliceViewer(squeeze(angle(h.refocusing_util)));error('asd')
            %make the cropped green function (for forward and backward field)
            params_truncated_green=h.parameters;
            params_truncated_green.size=h.parameters.size(:)...
                +[h.expected_RI_size(1) h.expected_RI_size(2) 0]'...
                +[h.parameters.RI_center(1) h.parameters.RI_center(2) 0]';
            warning('off','all');
            free_space_green=(truncated_green_plus(params_truncated_green));
            %free_space_green=(truncated_green_plus_v1(params_truncated_green));
            warning('on','all');
            free_space_green=free_space_green(...
                1-min(0,h.parameters.RI_center(1)):end-max(0,h.parameters.RI_center(1)),...
                1-min(0,h.parameters.RI_center(2)):end-max(0,h.parameters.RI_center(2)),:);
            free_space_green=circshift(free_space_green,[-h.parameters.RI_center(1) -h.parameters.RI_center(2) 0]);
            free_space_green=fftshift(ifftn(ifftshift(free_space_green)));
            h.kernel_trans=fftshift(fft2(ifftshift(conj(free_space_green))));
            h.kernel_ref=  fftshift(fft2(ifftshift((free_space_green))));
            
            h.kernel_ref=gather(h.kernel_ref);
            h.kernel_trans=gather(h.kernel_trans);
        end
        function matt=padd_RI2conv(h,matt)
            sz=size(matt);
            size_conv=h.parameters.size(1:2)'...
                +[h.expected_RI_size(1) h.expected_RI_size(2)]';
            
            add_start=-((floor(sz(1:2)'/2))-(floor(size_conv(:)/2)));
            add_end=size_conv(:)-sz(1:2)'-add_start(:);
            
            matt=padarray(matt,add_start,0,'pre');
            matt=padarray(matt,add_end,0,'post');
        end
        function matt=crop_conv2field(h,matt)
            sz=size(matt);
            ROI_start=(floor(sz(1:2)'/2)+1)-(floor(h.parameters.size(1:2)'/2));
            ROI_end=ROI_start+h.parameters.size(1:2)'-1;
            matt=matt(ROI_start(1):ROI_end(1),ROI_start(2):ROI_end(2),:,:);
        end
        function matt=crop_field2RI(h,matt)
            sz=size(matt);
            ROI_start=(floor(sz(1:2)'/2)+1)-(floor(h.parameters.size(1:2)'/2))...
                +[h.parameters.RI_center(1) h.parameters.RI_center(2)]';
            ROI_end=ROI_start+h.parameters.size(1:2)'-1;
            matt=matt(ROI_start(1):ROI_end(1),ROI_start(2):ROI_end(2),:,:);
        end
        function set_RI(h,RI)
            if ~isequal(size(RI)',h.expected_RI_size(:))
                error(['The refractiv index does not have the expected size : ' ...
                    num2str(h.expected_RI_size(1)) ' ' num2str(h.expected_RI_size(2)) ' ' num2str(h.expected_RI_size(3))]);
            end
            RI=single(RI);%single computation are faster
            
            set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            
            h.condition_RI();%modify the RI (add padding and boundary)
            h.init();%init the parameter for the forward model
        end
        function condition_RI(h)
            h.eps_imag = max(abs(RI2potential(h.RI(:),h.parameters.wavelength,h.parameters.RI_bg))).*1.01;
            step = abs(2*(2*pi*(h.parameters.RI_bg/h.parameters.wavelength))/h.eps_imag);
            h.pixel_step_size=round(step./(h.parameters.resolution));
            %add boundary to the RI
            h.ROI = h.create_boundary_RI(); %-CHANGED
            %update the size in the parameters
            h.V = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
            % h.V = h.V - 1i.*h.eps_imag; %-CHANGED
            if size(h.V,4)==1
                h.V = h.V - 1i.*h.eps_imag;
            else
                for j1 = 1:3
                    h.V(:,:,:,j1,j1) = h.V(:,:,:,j1,j1) - 1i.*h.eps_imag;
                end
            end
            
        end
        function ROI = create_boundary_RI(h)
            warning('choose a higher size boundary to a size which fft is fast ??');
            warning('allow to chose a threshold for remaining energy');
            warning('min boundary size at low RI ??');
            % Set boundary size & absorptivity
            if length(h.parameters.boundary_thickness) == 1
                error('Boundary in only one direction is not precise; enter "boundary_thickness" as an array of size 3 use size 0 for ciclic boundary');
            elseif length(h.parameters.boundary_thickness) == 3
                h.boundary_thickness_pixel = round((h.parameters.boundary_thickness*h.parameters.wavelength/h.parameters.RI_bg)./(h.parameters.resolution.*2));
            else
                error('h.boundary_thickness_pixel vector dimension should be 1 or 3.')
            end
            
            % Pad boundary
            if (h.parameters.use_GPU)
                h.RI = gpuArray(h.RI);
            end
            old_RI_size=size(h.RI);
            h.RI=padarray(h.RI,...
                [h.boundary_thickness_pixel(1) h.boundary_thickness_pixel(2) h.boundary_thickness_pixel(3)],...
                h.parameters.RI_bg);
            
            ROI = [...
                h.boundary_thickness_pixel(1)+1 h.boundary_thickness_pixel(1)+old_RI_size(1)...
                h.boundary_thickness_pixel(2)+1 h.boundary_thickness_pixel(2)+old_RI_size(2)...
                h.boundary_thickness_pixel(3)+1 h.boundary_thickness_pixel(3)+old_RI_size(3)];
            %ROI = [1 ZP0(1) 1 ZP0(2) 1 ZP0(3)];
            
            V_temp = RI2potential(h.RI,h.parameters.wavelength,h.parameters.RI_bg);
            
            h.attenuation_mask=1;
            for j1 = 1:3
                x=single(abs((1:size(V_temp,j1))-(floor(size(V_temp,j1)/2+1))+0.5)-0.5);x=circshift(x,-floor(size(V_temp,j1)/2));
                x=x/(h.boundary_thickness_pixel(j1)-0.5);
                %x=circshift(x,size(V_temp,j1)-round(h.boundary_thickness_pixel(j1)/2));
                val0=x;val0(abs(x)>=1)=1;val0=abs(val0);
                val0=1-val0;
                val0(val0>h.parameters.boundary_sharpness)=h.parameters.boundary_sharpness;
                val0=val0./h.parameters.boundary_sharpness;
                if h.boundary_thickness_pixel(j1)==0
                    val0(:)=0;
                end
                if j1 == 1
                    h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,[],1,1).*1);
                elseif j1 == 2
                    h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,1,[],1).*1);
                else
                    h.attenuation_mask=h.attenuation_mask.*(1-reshape(val0,1,1,[]).*1);
                end
            end
            %figure; plot(squeeze(h.attenuation_mask{end})),title('Axial boundary window attenuator strength');
            
            h.RI = potential2RI(V_temp,h.parameters.wavelength,h.parameters.RI_bg);
            if (h.parameters.use_GPU)
                h.RI=gather(h.RI);
                h.attenuation_mask=gather(h.attenuation_mask);
            end
        end
        function init(h)
            params_border=h.parameters;
            params_border.size=size(h.RI);
            warning('off','all');
            h.utility_border=DERIVE_OPTICAL_TOOL(params_border,h.parameters.use_GPU); % the utility for the space with border
            warning('on','all');
            
            if h.parameters.verbose && h.parameters.iterations_number>0
                warning('Best is to set iterations_number to -n for an automatic choice of this so that reflection to the ordern n-1 are taken in accound (transmission n=1, single reflection n=2, higher n=?)');
            end
            
            if h.parameters.use_GPU
                h.RI=single(gpuArray(h.RI));
            end
            h.pole_num=1;
            if h.parameters.vector_simulation
                h.pole_num=3;
            end
            
            shifted_coordinate=cell(3,1);
            if h.parameters.acyclic && h.boundary_thickness_pixel(1)==0 && h.boundary_thickness_pixel(2)==0
                %shift only in z h.parameters.acyclic
                shifted_coordinate{1}=h.utility_border.fourier_space.coor{1};
                shifted_coordinate{2}=h.utility_border.fourier_space.coor{2};
                shifted_coordinate{3}=h.utility_border.fourier_space.coor{3}+h.utility_border.fourier_space.res{3}/4;
            elseif h.parameters.acyclic
                %shift all by kres/4
                shifted_coordinate{1}=h.utility_border.fourier_space.coor{1}+h.utility_border.fourier_space.res{1}/4;
                shifted_coordinate{2}=h.utility_border.fourier_space.coor{2}+h.utility_border.fourier_space.res{2}/4;
                shifted_coordinate{3}=h.utility_border.fourier_space.coor{3}+h.utility_border.fourier_space.res{3}/4;
            else
                %no shift
                shifted_coordinate{1}=h.utility_border.fourier_space.coor{1};
                shifted_coordinate{2}=h.utility_border.fourier_space.coor{2};
                shifted_coordinate{3}=h.utility_border.fourier_space.coor{3};
            end
            
            h.rads=...
                (shifted_coordinate{1}./h.utility_border.k0_nm).*reshape([1 0 0],1,1,1,[])+...
                (shifted_coordinate{2}./h.utility_border.k0_nm).*reshape([0 1 0],1,1,1,[])+...
                (shifted_coordinate{3}./h.utility_border.k0_nm).*reshape([0 0 1],1,1,1,[]);
            %error('need to make true k/4 shift for rad !!!');
            
            h.green_absorbtion_correction=((2*pi*h.utility_border.k0_nm)^2)/((2*pi*h.utility_border.k0_nm)^2+1i.*h.eps_imag);
            
            step = abs(2*(2*pi*h.utility_border.k0_nm)/h.eps_imag);
            Bornmax_opt = ceil(norm(size(h.RI,1:3).*h.parameters.resolution) / step / 2 + 1)*2; % -CHANGED
            h.Bornmax = 0;
            
            if h.parameters.iterations_number==0
                error('set iterations_number to either a positive or negative value');
            elseif h.parameters.iterations_number<=0
                h.Bornmax=Bornmax_opt*abs(h.parameters.iterations_number);
            else
                h.Bornmax =h.parameters.iterations_number;
            end
            
            if h.parameters.verbose
                display(['number of step : ' num2str(h.Bornmax)])
                display(['step pixel size : ' num2str(h.pixel_step_size(3))])
            end
            
            h.eye_3=single(reshape(eye(3),1,1,1,3,3));
            if h.parameters.use_GPU
                h.eye_3=gpuArray(h.eye_3);
            end
            h.Greenp = 1 ./ (4*pi^2.*abs(...
                (shifted_coordinate{1}).^2 + ...
                (shifted_coordinate{2}).^2 + ...
                (shifted_coordinate{3}).^2 ...
                )-(2*pi*h.utility_border.k0_nm)^2-1i*h.eps_imag);
            
            %error('need to make a true k/4 shift !!!');
            
            if h.parameters.acyclic
                if h.boundary_thickness_pixel(1)==0 && h.boundary_thickness_pixel(2)==0
                    x=exp(-1i.*pi.*((1:size(h.V,3))-1)./size(h.V,3)/2);
                    %x=circshift(x,-round(h.boundary_thickness_pixel/2));
                    x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
                    h.phase_ramp=reshape(x,1,1,[]);
                else
                    for j1 = 1:3
                        x=single(exp(-1i.*pi.*((1:size(h.V,j1))-1)./size(h.V,j1)/2));
                        %x=circshift(x,-round(h.boundary_thickness_pixel(j1)/2));
                        x=x./x(floor(size(x,1)/2)+1,floor(size(x,2)/2)+1,floor(size(x,3)/2)+1);
                        if j1 == 1
                            h.phase_ramp=reshape(x,[],1,1);
                        elseif j1 == 2
                            h.phase_ramp=h.phase_ramp.*reshape(x,1,[],1);
                        else
                            h.phase_ramp=h.phase_ramp.*reshape(x,1,1,[]);
                        end
                    end
                end
            else
                h.phase_ramp=1;
            end
            
            if h.parameters.use_GPU
                
                h.rads = gpuArray(single(h.rads));
                h.Greenp = gpuArray(single(h.Greenp));
            end
            
            h.Greenp=ifftshift(ifftshift(ifftshift(h.Greenp,1),2),3);
            h.rads=ifftshift(ifftshift(ifftshift(h.rads,1),2),3);
            
            if h.parameters.verbose
                figure('units','normalized','outerposition',[0 0 1 1])
                colormap hot
            end
            
            h.RI=gather(h.RI);
            h.phase_ramp=gather(h.phase_ramp);
            h.rads = gather(single(h.rads));
            h.Greenp = gather(single(h.Greenp));
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
            
            for field_num=1:size(input_field,4)
                Field=h.solve_raw(input_field(:,:,:,field_num));
                %crop and remove near field (3D to 2D field)
                if h.parameters.return_3D
                    fields_3D(:,:,:,:,field_num)=gather(Field);
                end
                if h.parameters.return_transmission || h.parameters.return_reflection
                    potential=RI2potential(h.RI(h.ROI(1):h.ROI(2), h.ROI(3):h.ROI(4), h.ROI(5):h.ROI(6),:,:),h.parameters.wavelength,h.parameters.RI_bg);
                    
                    emitter_3D=h.padd_RI2conv(Field.*potential.*h.utility_border.dV);
                    emitter_3D=fftshift(fft2(ifftshift(emitter_3D)));
                end
                if h.parameters.return_transmission
                    %figure; orthosliceViewer(abs(h.kernel_trans));
                    if (h.parameters.use_GPU)
                        h.kernel_trans=gpuArray(h.kernel_trans);
                    end
                    %h.kernel_trans
                    field_trans = h.crop_conv2field(fftshift(ifft2(ifftshift(sum(emitter_3D.*h.kernel_trans,3)))));
                    field_trans=squeeze(field_trans);
%                     if size(field_trans,3)>1
%                         error('still not implemented');
%                     end
                    field_trans=fftshift(fft2(ifftshift(field_trans)));
                    field_trans=field_trans+input_field(:,:,:,field_num);
                    [field_trans] = h.transform_field_2D(field_trans);
                    field_trans=fftshift(ifft2(ifftshift(field_trans)));
                    fields_trans(:,:,:,field_num)=gather(squeeze(field_trans));
                    
                    h.kernel_trans=gather(h.kernel_trans);
                end
                if h.parameters.return_reflection
                    if (h.parameters.use_GPU)
                        h.kernel_ref=gpuArray(h.kernel_ref);
                    end
                    field_ref = h.crop_conv2field(fftshift(ifft2(ifftshift(sum(emitter_3D.*h.kernel_ref,3)))));
                    field_ref=squeeze(field_ref);
%                     if size(field_ref,3)>1
%                         error('still not implemented');
%                     end
                    field_ref=fftshift(fft2(ifftshift(field_ref)));
                    [field_ref] = h.transform_field_2D_reflection(field_ref);
                    field_ref=fftshift(ifft2(ifftshift(field_ref)));
                    fields_ref(:,:,:,field_num)=gather(squeeze(field_ref));
                    
                    h.kernel_ref=gather(h.kernel_ref);
                end
            end
            %gather to release gpu memory
            h.V=gather(h.V);
            h.attenuation_mask=gather(h.attenuation_mask);
            h.phase_ramp=gather(h.phase_ramp);
            h.rads = gather(single(h.rads));
            h.Greenp = gather(single(h.Greenp));
            h.eye_3 = gather(h.eye_3);
        end
        function Field=solve_raw(h,source)
            size_field=[size(h.V,1),size(h.V,2),size(h.V,3),h.pole_num];
            if (h.parameters.use_GPU)
                h.V=gpuArray(h.V);
                h.refocusing_util=gpuArray(h.refocusing_util);
                h.phase_ramp=gpuArray(h.phase_ramp);
                source=gpuArray(single(source));
                h.attenuation_mask=gpuArray(h.attenuation_mask);
                h.rads = gpuArray(h.rads);
                h.eye_3 = gpuArray(h.eye_3);
                h.Greenp = gpuArray(h.Greenp);
                psi = zeros(size_field,'single','gpuArray');
                PSI = zeros(size_field,'single','gpuArray');
                Field = zeros(size_field,'single','gpuArray');
                Field_n = zeros(size_field,'single','gpuArray');
            else
                psi = zeros(size_field,'single');
                PSI = zeros(size_field,'single');
                Field = zeros(size_field,'single');
                Field_n = zeros(size_field,'single');
            end
            
            source = (reshape(source, [size(source,1),size(source,2),1,size(source,3)]).*h.refocusing_util);
            h.refocusing_util=gather(h.refocusing_util);
            source = fftshift(ifft2(ifftshift(source)));
            source = h.crop_field2RI(source);
            
            incident_field = source;
            if size(h.RI,4)==1
                source = (h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6))+1i*h.eps_imag).*source;
            else
                source00 = source;
                source(:) = 0;
                for j1 = 1:3
                    source = source + (h.V(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),(h.ROI(5)):(h.ROI(6)),:,j1)+1i*h.eps_imag) .* source00(:,:,:,j1);
                end
            end
            clear source00
            
            for jj = 1:h.Bornmax
                %flip the relevant quantities
                if h.parameters.acyclic
                    h.Greenp=fft_flip(h.Greenp,[1 1 1],false);
                    if h.pole_num==3
                        h.rads=fft_flip(h.rads,[1 1 1],false);
                    end
                    h.phase_ramp=conj(h.phase_ramp);
                end
                
                %init other quantities
                PSI(:) = 0;
                psi(:) = 0;
                if any(jj == [1,2])
                   Field_n(:)=0; 
                end
                
                
                if any(jj == [1,2]) % s
                    psi(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) = (1i./ h.eps_imag).*source/2;
                else % gamma * E
                    if size(h.V,4) == 1
                        psi = (1i./h.eps_imag) .* Field_n .* h.V;
                    else
                        for j1 = 1:3
                            psi = psi + (1i./h.eps_imag) .* Field_n(:,:,:,j1) .* h.V(:,:,:,:,j1);
                        end
                    end
                end
                
                % G x
                for j2 = 1:h.pole_num
                    coeff_field=fftn(psi(:,:,:,j2).*h.phase_ramp);
                    if h.pole_num==3 %dyadic absorptive green function convolution
                        PSI = PSI + ((h.Greenp.*(h.eye_3(:,:,:,:,j2)-h.green_absorbtion_correction*(h.rads).*(h.rads(:,:,:,j2)))).* coeff_field);
                    else %dscalar absorptive green function convolution
                        PSI = (h.Greenp.*coeff_field);
                    end
                end
                for j1 = 1:h.pole_num
                    PSI(:,:,:,j1)=ifftn(PSI(:,:,:,j1)).*conj(h.phase_ramp);
                end
                if ~any(jj == [1,2])
                    Field_n = Field_n - psi;
                end
                if size(h.V,4) == 1
                    Field_n = Field_n + (h.V) .* PSI;
                else
                    for j1 = 1:3
                        Field_n = Field_n + (h.V(:,:,:,:,j1)) .* h.PSI(:,:,:,j1);
                    end
                end
                % Attenuation
                Field_n=Field_n.*h.attenuation_mask;
                % add the fields to the total field
                if jj==2
                    clear source;
                    temp=Field;
                end
                Field = Field + Field_n;
                if jj==3
                    Field_n=Field_n+temp;
                    clear temp;
                end
            end
            
            Field = ...
                Field(h.ROI(1):h.ROI(2),h.ROI(3):h.ROI(4),h.ROI(5):h.ROI(6),:,:) + incident_field;
            
            if h.parameters.verbose
                set(gcf,'color','w'), imagesc((abs(squeeze(Field(:,floor(size(Field,2)/2)+1,:))'))),axis image, title(['Iteration: ' num2str(jj) ' / ' num2str(h.Bornmax)]), colorbar, axis off,drawnow
                colormap hot
            end
        end   
    end
end


