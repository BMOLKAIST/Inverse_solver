

%f1fun = @(r) sinc(2*r); % our sinc includes pi
%f1fun = @(r)  besselj(1,gamma* pi * r) ;
f1fun = @(r) (r<1);%
range=10;
samples=4000;

hankler=discrete_hankel(range,samples,0,true);
H3=hankler.forward(f1fun(hankler.get_r()));


r1=linspace(0,range,samples);

[H,k,r,I,K,R,h]=dht(f1fun,range,samples);
[H_out]=idht(H,I,K,R);

[H_out]=idht(2.*pi.*besselj(1,k)./(k),I,K,R);

k2=linspace(0,max(k(:)),samples);
[H2,I2]=ht(f1fun(r1(:)),r1(:),k2);

figure; 
plot(r,H_out);

figure; 
plot(k,H,'.');
hold on;
plot(k2,H2);
hold on;
plot(gather(hankler.get_k()),gather(H3),':');
hold on;
plot(k2,2.*pi.*besselj(1,k2)./(k2),'--');
hold on;
%plot(k2,(2.*pi).*besselj(1,k2)./(k2));




