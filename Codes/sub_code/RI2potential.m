function potential=RI2potential(RI,wavelength,base_RI)
k0=1/wavelength;
k =2*pi*base_RI*k0;
potential = single(k^2*(RI.^2/base_RI^2-1));
end