classdef FIELD_GENERATOR < handle
    properties (SetAccess = private, Hidden = true)
        parameters;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            %SIMULATION PARAMETERS
            params.percentage_NA_usage=0.95;
            params.illumination_style='random';% can be random circular etc...
            params.illumination_number=10;
            params.illumination_pol=[];
            params.start_with_normal=true;
            
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FIELD_GENERATOR(params)
            h.parameters=params;
        end
        function output_field=get_fields(h)
            if h.parameters.vector_simulation
                output_field=zeros(h.parameters.size(1),h.parameters.size(2),2,h.parameters.illumination_number,'single');
            else
                output_field=zeros(h.parameters.size(1),h.parameters.size(2),1,h.parameters.illumination_number,'single');
            end
            warning('off','all');
            utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            for ill_num=1:h.parameters.illumination_number
                d1=1;
                d2=1;
                
                switch h.parameters.illumination_style
                    case 'random'
                        while(sqrt(d1.^2+d2.^2)>1)
                            d1=2*(rand-1/2);
                            d2=2*(rand-1/2);
                        end
                    case 'circle'
                        while(sqrt(d1.^2+d2.^2)>1)
                            dividand=1/(h.parameters.illumination_number);
                            if h.parameters.start_with_normal
                                dividand=1/(h.parameters.illumination_number-1);
                            end
                            d1=sin((ill_num-1)*2*pi*(dividand));
                            d2=cos((ill_num-1)*2*pi*(dividand));
                        end
                    otherwise
                        error("Unknown illumination style name")
                end
                %first angle should be normal
                if ill_num==1 && h.parameters.start_with_normal 
                    d1=0;
                    d2=0;
                end
                try
                    output_field(...
                        floor(size(output_field,1)/2)+1+round(d1.*h.parameters.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{1}),...
                        floor(size(output_field,2)/2)+1+round(d2.*h.parameters.percentage_NA_usage.*utility.kmax./utility.fourier_space.res{2}),...
                        1,ill_num)=1;
                catch
                    error("An error occured while creating the field. can the resolution support the specified NA ?");
                end
            end
            output_field=fftshift(fft2(ifftshift(output_field)));
            
        end
    end
end