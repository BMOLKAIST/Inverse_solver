function [pos1,pos2,pos3] = peak_subpixel_positioner(peak_img,param1,param2,param3)
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
    [phase_coorelation,max_pos]=max(abs(peak_img(:)));
    pos1=param1(max_pos);
    pos2=param2(max_pos);
    pos3=param3(max_pos);
    %the subpixel registration : https://ieeexplore.ieee.org/document/988953
    %then use fourier shift theorem to subpixel shift the image
        %corrdinate of the main peak
        [id1,id2,id3]=ind2sub(size(peak_img),max_pos);
        %corrdinate of the second main peak in each direction
        choice=[id1-1,id1+1];
        [~,choice_pos]=max([peak_img(id1-1,id2,id3),peak_img(id1+1,id2,id3)]);
        id1_2=choice(choice_pos);
        choice=[id2-1,id2+1];
        [~,choice_pos]=max([peak_img(id1,id2-1,id3),peak_img(id1,id2+1,id3)]);
        id2_2=choice(choice_pos);
        if(size(peak_img,3)>1)
            choice=[id3-1,id3+1];
            [~,choice_pos]=max([peak_img(id1,id2-1,id3),peak_img(id1,id2+1,id3)]);
            id3_2=choice(choice_pos);
        end
        d1=[1,1].*peak_img(id1_2,id2  ,id3  )./(peak_img(id1_2,id2  ,id3  )+[1,-1].*peak_img(id1,id2,id3));
        d2=[1,1].*peak_img(id1  ,id2_2,id3  )./(peak_img(id1  ,id2_2,id3  )+[1,-1].*peak_img(id1,id2,id3));
        if(size(peak_img,3)>1)
            d3=[1,1].*peak_img(id1  ,id2  ,id3_2)./(peak_img(id1  ,id2  ,id3_2)+[1,-1].*peak_img(id1,id2,id3));
        end
        
        %imaginary part is noise
        d1=real(d1);
        d2=real(d2);
        if(size(peak_img,3)>1)
            d3=real(d3);
        end
        %is between 1 and -1
        d1(abs(d1)>1)=0;
        d2(abs(d2)>1)=0;
        if(size(peak_img,3)>1)
            d3(abs(d3)>1)=0;
        end
        
        %has same sign as idX_2-idX
        d1=sign(id1_2-id1)*max(d1);
        d2=sign(id2_2-id2)*max(d2);
        if(size(peak_img,3)>1)
            d3=sign(id3_2-id3)*max(d3);
        end
        
        pos1=pos1+d1;
        pos2=pos2+d2;
        if(size(peak_img,3)>1)
            pos3=pos3+d3;
        end
    
end