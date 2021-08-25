function [param1,param2,param3,param4]=CPU_placement_finder_prepare(size1,size2,size3,limit)
    %size1-> size of the input along dimention1
    %size2-> size of the input along dimention2
    %size3-> size of the input along dimention3 (size3 = 1 if 2D data)
    %limit-> maximum diplacement between the two images
    %param1 ->grid of positions (dim1)       
    %param2 ->grid of positions (dim2)       
    %param3 ->grid of positions (dim3)       
    %param4 ->matrix of position to search 
    
    [param2,param1,param3] = meshgrid((single(1:size2)),(single(1:size1)),(single(1:size3)));
    centre1=floor(size(param1,1)/2)+1;
    centre2=floor(size(param2,2)/2)+1;
    centre3=floor(size(param3,3)/2)+1;
    param1=param1-centre1;
    param2=param2-centre2;
    param3=param3-centre3;
    param4=(abs(param1)<=limit).*(abs(param2)<=limit).*(abs(param3)<=limit);
    
end