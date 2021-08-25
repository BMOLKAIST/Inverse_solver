function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = pol_vectors(ZP,n_m,lambda,dx)

ZP=ZP(1:2);

k0=1/lambda; %[um-1, spatial frequency @ vacuum]
k =2*pi*n_m*k0; % [um-1, spatial wavenumber @ medium ]

kres=1./dx(1:2)./ZP(1:2);

K_1=single(2*pi*kres(1)/k*(-floor(ZP(1)/2):ZP(1)-floor(ZP(1)/2)-1));%normalised to diffraction limit k1
K_2=single(2*pi*kres(2)/k*(-floor(ZP(2)/2):ZP(1)-floor(ZP(2)/2)-1));%normalised to diffraction limit k2

K_1=reshape(K_1,[],1);
K_2=reshape(K_2,1,[]);

K_3=real(sqrt(1-(K_1.^2+K_2.^2)));
K_mask = ~(K_3==0);

Radial_2D=zeros(ZP(1),ZP(2),2,'single');
Perp_2D=zeros(ZP(1),ZP(2),2,'single');
norm_rad=sqrt(K_1.^2+K_2.^2);

temp1=K_1./norm_rad;
temp2=K_2./norm_rad;
temp1(norm_rad==0)=1;
temp2(norm_rad==0)=0;

Radial_2D(:,:,1)=temp1;
Radial_2D(:,:,2)=temp2;
clear temp1;
clear temp2;

Perp_2D(:,:,1)=Radial_2D(:,:,2);
Perp_2D(:,:,2)=-Radial_2D(:,:,1); 

Radial_3D=zeros(ZP(1),ZP(2),3,'single');
norm_sph=sqrt(K_1.^2+K_2.^2+K_3.^2);
Radial_3D(:,:,1)=K_1./norm_sph;
Radial_3D(:,:,2)=K_2./norm_sph;
Radial_3D(:,:,3)=K_3./norm_sph;

ewald_TanProj=sum(Radial_3D(:,:,1:2).*Radial_2D,3);
ewald_TanVec=zeros(ZP(1),ZP(2),3,'single');

ewald_TanVec(:,:,1:2)=Radial_2D(:,:,:);
ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
ewald_TanVec_norm(~K_mask)=1;
ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;

end