classdef OVERLAP_COUNTER < BACKWARD_SOLVER_RYTOV
    properties (SetAccess = protected, Hidden = true)
        
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            params=get_default_parameters@BACKWARD_SOLVER_RYTOV();
            %specific parameters
            
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=OVERLAP_COUNTER(params)
            h@BACKWARD_SOLVER_RYTOV(params);
        end
        function [overlap_3D,overlap_field]=get_overlap(h,input_field)
            h.solve(input_field,input_field);
            overlap_3D=h.overlap_3D;
            overlap_field=h.overlap_field;
        end
    end
end