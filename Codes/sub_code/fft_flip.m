function array=fft_flip(array,flip_bool, use_shift)
%use_shift is set to true if the array has the zero frequency centered
for ii = 1:length(flip_bool)
    if flip_bool(ii)
        array=flip(array,ii);
        if mod(size(array,ii),2)==0 || ~use_shift
            array=circshift(array,1,ii);
        end
    end
end
end
