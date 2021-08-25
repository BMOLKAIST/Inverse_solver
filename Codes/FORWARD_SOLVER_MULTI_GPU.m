classdef FORWARD_SOLVER_MULTI_GPU < FORWARD_SOLVER
    properties %(SetAccess = protected, Hidden = true)
        queue_out;
        queue_in;
        parallel_return;
        num_worker;
        
        sz_RI;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@FORWARD_SOLVER();
            %specific parameters
            params.subclass=@() error("Please define the forward solver to scale in parallel ~");
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FORWARD_SOLVER_MULTI_GPU(params)
            h@FORWARD_SOLVER(params);
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
                h.parallel_return{ii}=parfeval(@forward_executer,1,h.queue_out{ii},h.parameters);
            end
            for ii=1:h.num_worker
                [h.queue_in{ii},success]=poll(h.queue_out{ii},10);
                send(h.queue_in{ii},ii);
                if ~success
                    error('At least one thread did not respond');
                else
                    %display('ok');
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
        end
        function set_RI(h,RI)
            h.sz_RI=size(RI);
            %set_RI@FORWARD_SOLVER(h,RI);%call the parent class function to save the RI
            for ii=1:h.num_worker
               packet={1,RI}; 
               send(h.queue_in{ii},packet);
            end
        end
        function [fields_trans,fields_ref,fields_3D]=solve(h,input_field)
            d3_pol_num=1;
            d2_pol_num=1;
            if h.parameters.vector_simulation
               d3_pol_num=3;
               d2_pol_num=2; 
            end
            fields_trans=[];
            if h.parameters.return_transmission
                fields_trans=ones(size(input_field,1),size(input_field,2),d2_pol_num,size(input_field,4),'single');
            end
            fields_ref=[];
            if h.parameters.return_reflection
                fields_ref=ones(size(input_field,1),size(input_field,2),d2_pol_num,size(input_field,4),'single');
            end
            fields_3D=[];
            if h.parameters.return_3D
                fields_3D=ones(h.sz_RI(1),h.sz_RI(2),h.sz_RI(3),d3_pol_num,size(input_field,4),'single');
            end
            num_fields=size(input_field,4);
            %resquest
            st=1;
            for ii=1:h.num_worker
               nd=round(ii.*num_fields./h.num_worker);
               
               if nd>=st && st<=num_fields
                   %display([num2str(st) ':' num2str(nd)]);
                   send(h.queue_in{ii},{2,input_field(:,:,:,st:nd),ii});
                   
                   st=nd+1;
               end
            end
            %{ 
            for ii=1:h.num_worker
                fetchOutputs(h.parallel_return{ii})
            end
            %}
            %recieve
            st=1;
            for ii=1:h.num_worker
               nd=round(ii.*num_fields./h.num_worker);
               if nd>=st && st<=num_fields
                   [packet,success]=poll(h.queue_out{ii},99999);
                   if h.parameters.return_transmission
                       fields_trans(:,:,:,st:nd)=packet{1};
                   end
                   if h.parameters.return_reflection
                       fields_ref(:,:,:,st:nd)=packet{2};
                   end
                   if h.parameters.return_3D
                       fields_3D(:,:,:,:,st:nd)=packet{3};
                   end
                   st=nd+1;
               end
            end
            %end
        end
        
    end
end
function res=forward_executer(data_queue_out,params)
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
solver=params.subclass(params);
send(data_queue_out,true);
%performing requests
running=true;
while running
    [request,success]=poll(request_queue,10);
    switch request{1}
    case 0
        running=false;
    case 1
        solver.set_RI(request{2});
    case 2
        [fields_trans,fields_ref,fields_3D]=solver.solve(request{2});
        send(data_queue_out,{fields_trans,fields_ref,fields_3D});
    otherwise
        error('unknown command');
end
end
%returning
res=true;
end

