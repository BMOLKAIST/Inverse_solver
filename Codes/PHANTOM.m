classdef PHANTOM
    methods(Static)
        function params=get_default_parameters()
            params.name='blank';
            params.antialiasing=true;
            params.outer_size=[100 100 100];
            params.inner_size=[50 50 50];
            params.rotation_angles = [0 0 0]; % Euler rotation / Z -> Y -> Z % [degree]
        end
        function phantom=get(params)
            outer_size=params.outer_size;
            inner_size=params.inner_size;
            phantom=zeros(outer_size,'single');
            d1=single(reshape(single(1:outer_size(1)),[],1,1)-(floor(outer_size(1)/2)+1));
            d2=single(reshape(single(1:outer_size(2)),1,[],1)-(floor(outer_size(2)/2)+1));
            d3=single(reshape(single(1:outer_size(3)),1,1,[])-(floor(outer_size(3)/2)+1));
            tot_sample=1;
            if params.antialiasing
                tot_sample=8;
            end
            for sample_num=1:tot_sample
                sample_shift=[0 0 0];
                if params.antialiasing
                    switch sample_num
                        case 1
                            sample_shift=[1 1 1];
                        case 2
                            sample_shift=[-1 1 1];
                        case 3
                            sample_shift=[1 -1 1];
                        case 4
                            sample_shift=[-1 -1 1];
                        case 5
                            sample_shift=[1 1 -1];
                        case 6
                            sample_shift=[-1 1 -1];
                        case 7
                            sample_shift=[1 -1 -1];
                        case 8
                            sample_shift=[-1 -1 -1];
                        otherwise
                    end
                    sample_shift=(1/4).*sample_shift;
                end
                d1_norm= 2.*(d1+sample_shift(1))./inner_size(1);
                d2_norm= 2.*(d2+sample_shift(2))./inner_size(2);
                d3_norm= 2.*(d3+sample_shift(3))./inner_size(3);
                r_norm=sqrt(d1_norm.^2+d2_norm.^2+d3_norm.^2);
                switch params.name
                    case 'blank'
                        %nothing to do return blacnk
                    case 'bead'
                        %make a bead
                        phantom=phantom+single(r_norm<1);
                    case 'dimer'
                        %make a bead
                        phantom=phantom+circshift(single(r_norm<1), round([inner_size(2) 0 0]/2))+circshift(single(r_norm<1), round([-inner_size(2) 0 0]/2));
                        phantom(phantom>0.5) = 1;
                    case 'trimer'
                        %make a bead
                        radius = inner_size(2)/2 / sin(pi/3);
                        for j1 = 1:5
                            phantom = phantom + circshift(single(r_norm<1), round(radius*[cos(2*pi/3*(j1-1)) sin(2*pi/3*(j1-1)) 0]));
                        end
                        phantom(phantom>0.5) = 1;
                    case 'pentamer'
                        %make a bead
                        radius = inner_size(2)/2 / sin(pi/5);
                        for j1 = 1:5
                            phantom = phantom + circshift(single(r_norm<1), round(radius*[cos(2*pi/5*(j1-1)) sin(2*pi/5*(j1-1)) 0]));
                        end
                        phantom(phantom>0.5) = 1;
                    case 'RBC'
                        %make a bead
                        SPQR =  [0.193 14.3 38.9 4.57] ;
                        % Ref: Joowon Lim paper is wrong. Refer to
                        % https://pure.uva.nl/ws/files/4217415/52862_yurkin_thesis.pdf Eq. 179
                        Rho = sqrt(d1_norm.^2 + d2_norm.^2);
                        rbc_funct=@(Rho) ((-(SPQR(3)-2*SPQR(1).*(Rho.^2)) + real(sqrt((SPQR(3)-2*SPQR(1).*(Rho.^2)).^2-4*(Rho.^4-SPQR(2)*Rho.^2-SPQR(4))))) / 2);
                        sz_rbc=(SPQR(2)+sqrt(SPQR(2).^2+4.*SPQR(4)))/2;
                        sz_rbc=sqrt(sz_rbc);
                        phantom=phantom+ imclearborder(single(...
                            (d3_norm.*sz_rbc).^2 <= rbc_funct(Rho.*sz_rbc)...
                            ));
                    case 'SheppLogan'
                        phantom0 = phantom3d;
                        phantom(floor(end/2)+1-floor(size(phantom0,1)/2):floor(end/2)+size(phantom0,1)-floor(size(phantom0,1)/2),...
                            floor(end/2)+1-floor(size(phantom0,2)/2):floor(end/2)+size(phantom0,2)-floor(size(phantom0,2)/2),...
                            floor(end/2)+1-floor(size(phantom0,3)/2):floor(end/2)+size(phantom0,3)-floor(size(phantom0,3)/2)) = phantom0;
                    otherwise
                        error('Unknown phantom name')
                end
            end
            %normalise
            phantom=phantom-min(phantom(:));
            if max(phantom(:))~=0
                phantom=phantom./max(phantom(:));
            end
            if norm(params.rotation_angles)>0
                phantom_dummy = phantom * 0;
                phantom = imrotate3(phantom, params.rotation_angles(1), [0 0 1], 'crop');
                phantom = imrotate3(phantom, params.rotation_angles(2), [1 0 0], 'crop');
                phantom = imrotate3(phantom, params.rotation_angles(3), [0 0 1], 'crop');
                phantom_dummy(floor(end/2)+1-floor(size(phantom,1)/2):floor(end/2) + size(phantom,1) - floor(size(phantom,1)/2),...
                    floor(end/2)+1-floor(size(phantom,2)/2):floor(end/2) + size(phantom,2) - floor(size(phantom,2)/2),...
                    floor(end/2)+1-floor(size(phantom,3)/2):floor(end/2) + size(phantom,3) - floor(size(phantom,3)/2)) = phantom;
                phantom = phantom_dummy;
            end
        end
    end
end
