function RI=potential2RI(potential,wavelength,base_RI)
k0=1/wavelength;
k =2*pi*base_RI*k0;
RI=single(base_RI*sqrt(1+potential./(k.^2)));
end