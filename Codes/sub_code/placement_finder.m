function [pos1,pos2,pos3,phase_coorelation,pearson_coefficient] = placement_finder(mat1,mat2,param1,param2,param3,param4,sub_pixel_bool,pearson_bool)
    %mat1 ->matrix number one
    %mat2 ->matrix number2
    
    %param1 ->grid of positions (dim1)       (use placement_finder_prepare to find it)
    %param2 ->grid of positions (dim2)       (use placement_finder_prepare to find it)
    %param3 ->grid of positions (dim3)       (use placement_finder_prepare to find it)
    %param4 ->matrix of position to search   (use placement_finder_prepare to find it)
    %pos1 ->the difference in position between the two matrix along the first diemention
    %pos2 ->the difference in position between the two matrix along the second diemention
    %pos3 ->the difference in position between the two matrix along the fird diemention
    %correlation -> the phase correlation at that point (sign of how much similar the two are (can be use as a weight wen positioning several pages))
    %sub_pixel_bool -> use subpixel positioning
    
    %the phase correlation method : https://en.wikipedia.org/wiki/Phase_correlation
    mat1_2=fftn(ifftshift(mat1));
    mat2_2=fftn(ifftshift(mat2));
    mat1_2(1,1,1)=0;%put the mean to zero
    mat2_2(1,1,1)=0;%put the mean to zero
    %the croscorelation
    cross_correleation=(mat1_2.*conj(mat2_2)); 
    %the cross corelation phase
    cross_correleation(cross_correleation==0)=1;%to avoid division by 0
    cross_correleation=cross_correleation./abs(cross_correleation);
    %crop to the region of interest
    cross_correleation=fftshift(ifftn(cross_correleation)).*param4;%limit the search of the position to the limit
    %get maximum
    [phase_coorelation,max_pos]=max(abs(cross_correleation(:)));
    pos1=param1(max_pos);
    pos2=param2(max_pos);
    pos3=param3(max_pos);
    %the subpixel registration : https://ieeexplore.ieee.org/document/988953
    %then use fourier shift theorem to subpixel shift the image
    if sub_pixel_bool
        %corrdinate of the main peak
        [id1,id2,id3]=ind2sub(size(cross_correleation),max_pos);
        %corrdinate of the second main peak in each direction
        choice=[id1-1,id1+1];
        [~,choice_pos]=max([cross_correleation(id1-1,id2,id3),cross_correleation(id1+1,id2,id3)]);
        id1_2=choice(choice_pos);
        choice=[id2-1,id2+1];
        [~,choice_pos]=max([cross_correleation(id1,id2-1,id3),cross_correleation(id1,id2+1,id3)]);
        id2_2=choice(choice_pos);
        if(size(mat1,3)>1)
            choice=[id3-1,id3+1];
            [~,choice_pos]=max([cross_correleation(id1,id2-1,id3),cross_correleation(id1,id2+1,id3)]);
            id3_2=choice(choice_pos);
        end
        d1=[1,1].*cross_correleation(id1_2,id2  ,id3  )./(cross_correleation(id1_2,id2  ,id3  )+[1,-1].*cross_correleation(id1,id2,id3));
        d2=[1,1].*cross_correleation(id1  ,id2_2,id3  )./(cross_correleation(id1  ,id2_2,id3  )+[1,-1].*cross_correleation(id1,id2,id3));
        if(size(mat1,3)>1)
            d3=[1,1].*cross_correleation(id1  ,id2  ,id3_2)./(cross_correleation(id1  ,id2  ,id3_2)+[1,-1].*cross_correleation(id1,id2,id3));
        end
        
        %imaginary part is noise
        d1=real(d1);
        d2=real(d2);
        if(size(mat1,3)>1)
            d3=real(d3);
        end
        %is between 1 and -1
        d1(abs(d1)>1)=0;
        d2(abs(d2)>1)=0;
        if(size(mat1,3)>1)
            d3(abs(d3)>1)=0;
        end
        
        %has same sign as idX_2-idX
        d1=sign(id1_2-id1)*max(d1);
        d2=sign(id2_2-id2)*max(d2);
        if(size(mat1,3)>1)
            d3=sign(id3_2-id3)*max(d3);
        end
        
        pos1=pos1+d1;
        pos2=pos2+d2;
        if(size(mat1,3)>1)
            pos3=pos3+d3;
        end
    end
    %compute the pearson coefficient https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    pearson_coefficient=1;%init the coefficient
    if pearson_bool
        int_pos1=round(pos1);
        int_pos2=round(pos2);
        int_pos3=round(pos3);
        d_pos1=pos1-int_pos1;
        d_pos2=pos2-int_pos2;
        d_pos3=pos3-int_pos3;
        %cut to the overlapping area
        O2=mat2(   1-(int_pos1<0)*int_pos1:size(mat2,1)-(int_pos1>0)*int_pos1,...
                   1-(int_pos2<0)*int_pos2:size(mat2,2)-(int_pos2>0)*int_pos2,...
                   1-(int_pos3<0)*int_pos3:size(mat2,3)-(int_pos3>0)*int_pos3);
        O1=mat1(   1+(int_pos1>0)*int_pos1:size(mat2,1)+(int_pos1<0)*int_pos1,...
                   1+(int_pos2>0)*int_pos2:size(mat2,2)+(int_pos2<0)*int_pos2,...
                   1+(int_pos3>0)*int_pos3:size(mat2,3)+(int_pos3<0)*int_pos3);
        if sub_pixel_bool
            O1=real(fourier_shift(O1,-d_pos1,-d_pos2,-d_pos3));
            %crop 1 more pixel
            if(size(mat1,3)>4)%4 is arbitrary just to avoid cropping when there is almost no data
            O1=O1(2:size(O1,1)-1,2:size(O1,2)-1,2:size(O1,3)-1);
            O2=O2(2:size(O2,1)-1,2:size(O2,2)-1,2:size(O2,3)-1);
            end
        end
        pearson_coefficient=corrcoef(O1(:),O2(:));
        pearson_coefficient=pearson_coefficient(1,2);
    end
    
end