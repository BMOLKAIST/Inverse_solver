classdef BACKWARD_SOLVER_SINGLE < BACKWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
%         utility;
        forward_solver;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@BACKWARD_SOLVER();
            %specific parameters
            params.forward_solver= @(x) FORWARD_SOLVER(x);
            params.forward_solver_parameters= FORWARD_SOLVER.get_default_parameters();
            params.init_solver=BACKWARD_SOLVER(BACKWARD_SOLVER.get_default_parameters());
            params.step=0.01;%0.01;0.01;%0.01;
            params.tv_param=0.001;%0.1;
            params.use_non_negativity=false;
            params.nmin = 1.336;
            params.nmax = 1.6;
            params.kappamax = 0; % imaginary RI
            params.inner_itt = 100; % imaginary RI
            params.itter_max = 100; % imaginary RI
            params.num_scan_per_iteration = 0; % 0 -> every scan is used
            params.verbose = true;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=BACKWARD_SOLVER_SINGLE(params)
            %do not set the init solver it is for porent class compatibility
            h@BACKWARD_SOLVER(params);
            if ~exist('init_solver','var')
                init_solver=true;
            end
            if init_solver
                h.forward_solver=h.parameters.forward_solver(h.parameters.forward_solver_parameters);
            end
        end
        function [RI]=solve(h,input_field,output_field)
            
            [RI, mask]=(h.parameters.init_solver.solve(input_field,output_field));
            RI = single(RI);
            mask = ifftshift(ifftshift(ifftshift(mask~=0)));
           
            err_list=[];
            
            dirichlet_boundary=false;
            use_gpu=true;

            alpha=1/h.parameters.step;
            s_n=0;
            
            t_n=0;
            t_np=1;
            
            u_n=(RI.^2 / h.parameters.RI_bg^2 - 1);
            x_n=(RI.^2 / h.parameters.RI_bg^2 - 1);
            c_n=0;
            c_np=Inf;
            
            if mod(size(RI),2)==0
                error('need to be odd size');
            end
            if h.parameters.verbose
                close all
                f1=figure(1);
                f2=figure(2);
                f3=figure(3);
                %{
                f4=figure(4);
                f5=figure(5);
                f6=figure(6);
                f7=figure(7);
                %}
            end
            
            Vmin = (h.parameters.nmin^2 - h.parameters.kappamax^2) / h.parameters.RI_bg^2 - 1;
            Vmax = h.parameters.nmax^2 / h.parameters.RI_bg^2 - 1;
            Vimag_max = 2 * h.parameters.kappamax *h.parameters.nmax / h.parameters.RI_bg^2;
            
            ORytov = fftn(RI.^2 / h.parameters.RI_bg^2 - 1) .* mask;
            
%             figure,orthosliceViewer(abs(gather(mask))), error
            
            for ii=1:h.parameters.itter_max
                
                display(['itter : ' num2str(ii)]);
                
                tic;
                t_n=t_np;
                c_n=c_np;
                
                gradient_RI = ifftn(fftn(RI.^2 / h.parameters.RI_bg^2 - 1).*mask - ORytov);
                
%                 size(u_n)
%                 size(gradient_RI)
                s_n=TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
                    Vmin, Vmax, Vimag_max, h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
%                 s_n=TV_FISTA_inner(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
                t_np=(1+sqrt(1+4*t_n^2))/2;
                u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                x_n=s_n;
                RI=u_n;
                RI=h.parameters.RI_bg *sqrt(RI+1);
                
                toc;
                
                if h.parameters.verbose
                    set(0, 'currentfigure', f1);
                    imagesc(real(squeeze(RI(round(size(RI,1)/2),:,:,1,1))));colorbar; axis image;title('RI (xz)')
                    set(0, 'currentfigure', f2);
                    plot(err_list);title('Cost function')
                    set(0, 'currentfigure', f3);
                    semilogy((err_list));title('Cost function (log)')
                    %{
                    set(0, 'currentfigure', f4);
                    imagesc([abs(squeeze(trans_source(:,:,1,[1]))) squeeze(abs(output_field(:,:,1,[1]))) squeeze(abs(trans_source(:,:,1,1)-output_field(:,:,1,1)))]); axis image;title('Abs (predicted / experimental / delta)'),colorbar
                    set(0, 'currentfigure', f5);
                    imagesc([abs(squeeze(trans_source(:,:,1,[end]))) squeeze(abs(output_field(:,:,1,[scan_list(end)]))) squeeze(abs(trans_source(:,:,1,[end])-output_field(:,:,1,[scan_list(end)])))]); axis image;title('Abs (predicted / experimental / delta)'),colorbar
                    set(0, 'currentfigure', f6);
                    imagesc([squeeze(angle(trans_source(:,:,1,[1]))) squeeze(angle(output_field(:,:,1,[scan_list(1)]))) angle(trans_source(:,:,1,1)./output_field(:,:,1,1))]);axis image;title('Angle (predicted / experimental)'),colorbar
                    set(0, 'currentfigure', f7);
                    imagesc([angle(trans_source(:,:,1,end)) angle(output_field(:,:,1,scan_list(end))) angle(trans_source(:,:,1,end)./output_field(:,:,1,scan_list(end)))]);axis image;title('Angle (predicted / experimental)'),colorbar
                    %}
                    drawnow;
                end
                
                
            end
            
            RI=gather(RI);
            
            
            warning('add kz ? also because of the abbe sine is it sqrt of kz ??')
            warning('implement cuda inner itteration tv');
            warning('implement mfista');
            
        end
    end
end