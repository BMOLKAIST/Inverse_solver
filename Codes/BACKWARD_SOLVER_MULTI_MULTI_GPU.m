classdef BACKWARD_SOLVER_MULTI_MULTI_GPU < BACKWARD_SOLVER_MULTI
    properties (SetAccess = protected, Hidden = true)
        queue_out;
        queue_in;
        parallel_return;
        num_worker;
    end
    methods(Static)
        
    end
    methods
        function h=BACKWARD_SOLVER_MULTI_MULTI_GPU(params)
            h@BACKWARD_SOLVER_MULTI(params,false);
            h.num_worker=gpuDeviceCount("available");
            if h.num_worker==0
               error('No gpu available'); 
            end
            %start the parallel pool
            curr_pool=gcp('nocreate');
            if length(curr_pool)>0
                if curr_pool.NumWorkers<(h.num_worker+1);
                    delete(gcp('nocreate'));
                    parpool(h.num_worker+1);
                end
            else
                parpool(h.num_worker+1);
            end
            display(['Using ' num2str(h.num_worker) ' GPU']);
            h.parallel_return=cell(h.num_worker,1);
            h.queue_in=cell(h.num_worker,1);
            h.queue_out=cell(h.num_worker,1);
            for ii=1:h.num_worker
                h.queue_out{ii}=parallel.pool.PollableDataQueue;
                h.parallel_return{ii}=parfeval(@backward_executer,1,h.queue_out{ii},h.parameters);
            end
            for ii=1:h.num_worker
                [h.queue_in{ii},success]=poll(h.queue_out{ii},10);
                send(h.queue_in{ii},ii);
                if ~success
                    error('At least one thread did not respond');
                else
                    display('ok');
                end
            end
            %{
            for ii=1:h.num_worker
                fetchOutputs(h.parallel_return{ii})
            end
            %}
            for ii=1:h.num_worker
                [success2,success]=poll(h.queue_out{ii},30);
                if ~success || ~success2
                    error('At least one thread did not respond');
                else
                    %display(['GPU ' num2str(ii) ' completed']);
                end
            end
        end
        function delete(h)
            for ii=1:h.num_worker
               packet={0,[]}; 
               send(h.queue_in{ii},packet);
            end
            delete(gcp('nocreate'));%since the sim takes long anyway better stop it to avoid blockage later if bug occured
        end
        function [gradient_RI,err]=get_gradiant(h,RI,input_field,output_field)
            
            if h.parameters.num_scan_per_iteration == 0
                scan_list = 1:size(input_field,4);
            else
                scan_list = unique([1 randperm(size(input_field,4),h.parameters.num_scan_per_iteration)]);
                scan_list = scan_list(1:h.parameters.num_scan_per_iteration);
                %scan_list
            end
            
            num_fields=length(scan_list);
            %resquest
            st=1;
            for ii=1:h.num_worker
               nd=round(ii.*num_fields./h.num_worker);
               
               if nd>=st && st<=num_fields
                   %display([num2str(st) ':' num2str(nd)]);
                   send(h.queue_in{ii},{1,RI,input_field(:,:,:,scan_list(st:nd)),output_field(:,:,:,scan_list(st:nd))});
                   
                   st=nd+1;
               end
            end
            %{ 
            for ii=1:h.num_worker
                fetchOutputs(h.parallel_return{ii})
            end
            %}
            %recieve
            gradient_RI=0;
            err=0;
            
            st=1;
            for ii=1:h.num_worker
               nd=round(ii.*num_fields./h.num_worker);
               if nd>=st && st<=num_fields
                   [packet,success]=poll(h.queue_out{ii},99999);
                   
                   sub_f_num=nd-st+1;
                   %size(packet{1})
                   %size(packet{2})
                   %sub_f_num
                   gradient_RI=gradient_RI+packet{1}*sub_f_num;
                   err=err+packet{2}*sub_f_num;
                   
                   st=nd+1;
               end
            end
            gradient_RI=gradient_RI./num_fields;
            err=err./num_fields;
        end
    end
end

function res=backward_executer(data_queue_out,params)
%say it is ready by sending a queue for later communication
request_queue=parallel.pool.PollableDataQueue;
send(data_queue_out,request_queue);
%select the gpu
[gpu_num,success]=poll(request_queue,10);
if ~success
    error('Could not find a gpu');
end
gpuDevice(gpu_num);
%build the solver
solver=params.forward_solver(params.forward_solver_parameters);
send(data_queue_out,true);
%performing requests
running=true;
while running
    [request,success]=poll(request_queue,10);
    switch request{1}
    case 0
        running=false;
    case 1
        [grad,err]=BACKWARD_SOLVER_MULTI_MULTI_GPU.get_gradiant_static(request{2},request{3},request{4},solver);
        send(data_queue_out,{grad,err});
    otherwise
        error('unknown command');
end
end
%returning
res=true;
end

