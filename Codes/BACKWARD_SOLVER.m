classdef BACKWARD_SOLVER < handle
    properties (SetAccess = protected, Hidden = true)
        parameters;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            %SOLVER PARAMETERS
            
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=BACKWARD_SOLVER(params)
            h.parameters=params;
        end
        function [RI]=solve(h,input_field,output_field)
            error("You need to specify the backward solver to solve");
            RI=[];
        end
    end
end