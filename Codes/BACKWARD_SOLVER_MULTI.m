classdef BACKWARD_SOLVER_MULTI < BACKWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        forward_solver;
        overlap_count;
        filter;
        RI_inter;
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
            params.nmin = -inf;%1.336;
            params.nmax = inf;%1.6;
            params.kappamax = 0; % imaginary RI
            params.inner_itt = 100; % imaginary RI
            params.itter_max = 100; % imaginary RI
            params.num_scan_per_iteration = 0; % 0 -> every scan is used
            params.verbose = true;
            %params.filter_by_count=false;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
        function [gradient_RI,err]=get_gradiant_static(RI,input_field,output_field,forward_solver)
            warning ('off','all');
            forward_solver.set_RI(RI);
            
            [trans_source,~,source_3D]=forward_solver.solve( input_field);
            %[source_3D,trans_source,~]
            diff_field=(trans_source-output_field);
            err=sum(abs(diff_field).^2,'all');
            %{
            if h.parameters.filter_by_count
                diff_field=fftshift(fft2(ifftshift(diff_field)));
                diff_field=diff_field.*reshape(h.filter,size(h.filter,1),size(h.filter,2),1,[]).*size(diff_field,4)/5;
                diff_field=fftshift(ifft2(ifftshift(diff_field)));
            end
            %}
            %err_list(end)
            %backpropagate
            warning ('off','all');
            forward_solver.set_RI(conj(flip(RI,3)));
            [~,~,diff_3D]=forward_solver.solve(conj(diff_field));
            diff_3D=flip(conj(diff_3D),3);
            warning ('on','all');
            %recon_RI=(recon_RI.^2 / n_m^2 - 1);
            gradient_RI=-(1i.*((mean(conj(source_3D).*diff_3D,[4 5]))));
        end
    end
    methods
        
        function h=BACKWARD_SOLVER_MULTI(params,init_solver)
            %do not set the init solver it is for porent class compatibility
            h@BACKWARD_SOLVER(params);
            if ~h.parameters.forward_solver_parameters.return_transmission || ...
                    ~h.parameters.forward_solver_parameters.return_reflection || ...
                    ~h.parameters.forward_solver_parameters.return_3D
                h.parameters.forward_solver_parameters
                error('need to set all return parameter to true in the forward solver');
                
            end
            if ~exist('init_solver','var')
                init_solver=true;
            end
            if init_solver
                h.forward_solver=h.parameters.forward_solver(h.parameters.forward_solver_parameters);
            end
            %{
            if h.parameters.filter_by_count
                h.overlap_count=OVERLAP_COUNTER(OVERLAP_COUNTER.get_default_parameters(params));
            end
            %}
        end
        function [gradient_RI,err]=get_gradiant(h,RI,input_field,output_field)
            
            if h.parameters.num_scan_per_iteration == 0
                scan_list = 1:size(input_field,4);
            else
                scan_list = unique([1 randperm(size(input_field,4),h.parameters.num_scan_per_iteration)]);
                scan_list = scan_list(1:h.parameters.num_scan_per_iteration);
                %scan_list
            end
            
            [gradient_RI,err]=h.get_gradiant_static(RI,input_field(:,:,:,scan_list,:),output_field(:,:,:,scan_list,:),h.forward_solver);
        end
        function [RI]=solve(h,input_field,output_field)
            
            RI=single(h.parameters.init_solver.solve(input_field,output_field));
            %{
            if h.parameters.filter_by_count
                [~,h.filter]=h.overlap_count.get_overlap(input_field);
                h.filter(h.filter>0)=1./h.filter(h.filter>0);
            end
            %}
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
            
            if mod(size(RI,3),2)==0
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
            
            for ii=1:h.parameters.itter_max
                
                display(['itter : ' num2str(ii)]);
                
                tic;
                
                [gradient_RI,err_list(end+1)]=h.get_gradiant(RI,input_field,output_field);
                %gradient_RI=real(gradient_RI);
                %MFISTA
                
                t_n=t_np;
                c_n=c_np;
                
%                 size(u_n)
%                 size(gradient_RI)
                s_n=TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
                    Vmin, Vmax, Vimag_max, h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
%                 s_n=TV_FISTA_inner(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
                s_n=gather(s_n);
                t_np=(1+sqrt(1+4*t_n^2))/2;
                u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                x_n=s_n;
                RI=u_n;
                RI=h.parameters.RI_bg *sqrt(RI+1);
                
                h.RI_inter=RI;
                
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