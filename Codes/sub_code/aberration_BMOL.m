%% 1. Initialize
% %{
clear

%%%%%%%% if images are stored in one tiff file

% cd('E:\aberration DATA\usaf')
% info = imfinfo('raw_bg.tiff'); % interferograms
% sz = length(info); % number of angle scan
% info = info(1); % use first image infomration

%%%%%%%%% if images are stored as separate files

spPath = '\\143.248.14.27\Complete_Archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\SiO2_mesh_sp\set000001';
bgPath = '\\143.248.14.27\Complete_Archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\SiO2_mesh_bg\set000001';
% spPath = '\\143.248.14.27\Complete_Archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\RBC_mesh_sp\set000001';
% bgPath = '\\143.248.14.27\Complete_Archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\RBC_mesh_bg\set000001';

cd(spPath)
list = dir([spPath,'\ODT_*']);
info = imfinfo(list(1).name); 
sz = length(list);

% sy = info(1).Height; sx = info(1).Width; % width and height to crop
% sy = 1024; sx = 1024; 
sy = 512; sx = 512;

if sy > info(1).Height || sx > info(1).Width
    disp('crop size is out of bound. check sx, sy', 'Error','error')
    return
end

switch pwd % for path dependent variables
    
    case 'E:\aberration DATA\usaf'
    % add more case for different data paths
        
%         cx = floor(info(1).Width/2+1); cy = floor(info(1).Height/2+1); % the center of the cropped FoV   
        cx = 1130; cy = 950;
        
    % define naObj, naCond and mag if needed
    % defuault values are 0.75, 0.57 and 55.55
    % once defined, default values are overwritten (see below)
    
    case '\\143.248.14.27\complete_archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\SiO2_mesh_sp\set000001'
        
        cx = 320; cy = 335;
%         cx = floor(info(1).Width/2+1); cy = floor(info(1).Height/2+1);
        naObj = 1.1;
        naCond = 1.1;
        mag = 100;
        
    case '\\143.248.14.27\complete_archive2\Moosung Lee\ODT_mesh_for_Herve_Yoonseok\RBC_mesh_sp\set000001'
        cx = 363; cy = 385;
%         cx = floor(info(1).Width/2+1); cy = floor(info(1).Height/2+1);
        naObj = 1.1;
        naCond = 1.1;
        mag = 100;        
        
end
      
crop = [cy-floor(sy/2);cy+ceil(sy/2)-1;cx-floor(sx/2);cx+ceil(sx/2)-1];

if min(crop(:)) < 0 || crop(2) > info(1).Height || crop(4) > info(1).Width
   disp('crop range out of bound. check cx, cy'), return 
end

% % spTemp = double(imread('raw_tomo1.tiff',readList(1)));
% spTemp = double(imread([spPath,'\',list(1).name]));
% spTemp = spTemp(crop(1):crop(2),crop(3):crop(4));
% figure(101), imagesc(spTemp), axis image, colorbar, return

pix = 5.5; % the pixel pitch of a camera in um (5.5 um for Lt425)

if ~exist('naObj','var')
    naObj = 0.75;
    naCond = 0.57;
end

if ~exist('mag','var')
    mag = 20/180*500;
end

% lb = 0.457; % wavelength
lb = 0.532;

fovX = pix*sx/mag; fovY = pix*sy/mag;
naX = naObj/lb*fovX; naY = naObj/lb*fovY;

naY = ceil(naY);
naX = ceil(naX);
sy2 = 2*naY; % size to crop Fourier space
sx2 = 2*naX;

normalIdx = 1; % index for the normal incident angle

readList = 1:1:sz; % index of images to use
readList = readList(readList~=normalIdx);
readList = [normalIdx, readList];

sz = length(readList);

fE = zeros(sy2,sx2,sz); % field in Fourier space
f_dy = zeros(sz,1); f_dx = zeros(sz,1); % peak position w.r.t the normal peak

% clear info % to save memory

%% 2. Field retrieval

for idx = 1:sz
% for idx = 211
  
disp(['Field retrieval : ',num2str(idx), ' / ', num2str(sz)])

% spTemp = double(imread('raw_tomo1.tiff',readList(idx)));
% bgTemp = double(imread('raw_bg.tiff',readList(idx)));

spTemp = double(imread([spPath,'\',list(readList(idx)).name]));
bgTemp = double(imread([bgPath,'\',list(readList(idx)).name]));

spTemp = spTemp(crop(1):crop(2),crop(3):crop(4));
bgTemp = bgTemp(crop(1):crop(2),crop(3):crop(4));

fspTemp = fftshift(fft2(ifftshift(spTemp)));
fbgTemp = fftshift(fft2(ifftshift(bgTemp)));

if idx == 1 % find DC peak for the normal incident angle
    [f_dy_ref,f_dx_ref] = find(abs(fbgTemp(1:round(0.4*sy),1:round(0.4*sx))) ...
        == abs(max(max(fbgTemp(1:round(0.4*sy),1:round(0.4*sx))))));
end

fspTemp = fspTemp(f_dy_ref-floor(naY):f_dy_ref+ceil(naY)-1, ...
    f_dx_ref-floor(naX):f_dx_ref+ceil(naX)-1);
fspTemp = fspTemp.*~mk_ellipse(naX,naY,sx2,sy2);

fbgTemp = fbgTemp(f_dy_ref-floor(naY):f_dy_ref+ceil(naY)-1, ...
    f_dx_ref-floor(naX):f_dx_ref+ceil(naX)-1);
fbgTemp = fbgTemp.*~mk_ellipse(naX,naY,sx2,sy2);

spTemp = fftshift(ifft2(ifftshift(fspTemp)));
bgTemp = fftshift(ifft2(ifftshift(fbgTemp)));

[f_dy(idx),f_dx(idx)] = find(abs(fbgTemp) == abs(max(fbgTemp(:))));

ETemp = spTemp./bgTemp;
ETemp = conj(ETemp); % this flips f_dy f_dy!!!!!!!

fE(:,:,idx) = fftshift(fft2(ifftshift(ETemp)));

end

% f_dy = (f_dy-f_dy(1));
% f_dx = (f_dx-f_dx(1));
f_dy = -(f_dy-f_dy(1)); % because of the conjugation
f_dx = -(f_dx-f_dx(1));

figure(201), imagesc(abs(bgTemp)), axis image, colorbar
figure(202), imagesc(abs(spTemp)), axis image, colorbar
figure(204), imagesc(log(abs(fftshift(fft2(ifftshift(ETemp)))))), axis image
% figure(203), plot(f_dx,f_dy,'r*'), axis image
% return

clear spTemp bgTemp fbgTemp fspTemp ETemp % to save memory

%}
%% 3. Stitch Pupil

% Fourier space mask. the smaller the better
% fMask = ~mk_ellipse(sx2/30,sy2/30,sx2,sy2); 
fMask = ~mk_ellipse(sx2/25,sy2/25,sx2,sy2);
% fMask = ~mk_ellipse(sx2/10,sy2/10,sx2,sy2);

% figure(301), imagesc(log(abs(fE(:,:,1).*fMask))), axis image, return

fdp =  fE.*conj(fE(:,:,1)); 
fdp = exp(1i*angle(fdp)).*fMask;

pPhase = zeros(sy2,sx2,sz);

for ii = 1:sz
    disp(['Stitcing pupil : ', num2str(ii),' / ', num2str(sz)])
    pPhase(:,:,ii) = circshift(fdp(:,:,ii),[f_dy(ii),f_dx(ii)]);
    
end

pPhase = pPhase.*~mk_ellipse(naX,naY,sx2,sy2);

% clear fdp

figure(302), 
subplot(1,3,1), imagesc(abs(fMask)), colorbar, axis image
subplot(1,3,2), imagesc(angle(sum(pPhase,3))), colorbar, axis image
subplot(1,3,3), imagesc(abs(sum(pPhase,3))), colorbar, axis image
% return

%% 4. Relative Phase Correction
% clearvars -except rr sz pPhase fE f_dy f_dx

pPhase2 = pPhase;
[~,seq] = sort(sqrt((f_dx).^2+(f_dy).^2));

pStep = 5; % 3 takes longer time
phiList = linspace(0,2*pi,pStep).';

ft = fittype( 'a*cos(x-b)+c', 'independent', 'x', 'dependent', 'y' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% opts.Display = 'Off';
    
pPhase2Temp = pPhase;
sumTemp = 0;

for ii = 2:sz
    disp(['Phase matching : ', num2str(ii),' / ', num2str(sz)])
    
    temp = zeros(pStep,1);    
    sumTemp = sumTemp + pPhase2Temp(:,:,seq(ii-1));
    
%     figure(1), imagesc(abs(sum(sumTemp,3))), axis image, colorbar, pause()
    
    for pIter = 1:pStep
        
        pPhaseTemp = pPhase2(:,:,seq(ii))*exp(1i*phiList(pIter));
        temp(pIter) = sum(abs( sumTemp + pPhaseTemp ).^2, [1,2] );
%         temp(pIter) = sum(abs( exp(1i*angle(sumTemp)) + exp(1i*angle(pPhaseTemp)) ).^2, [1,2] );
    end
    
    opts.StartPoint = [(max(temp)-min(temp))/2 0 mean(temp)];
    [fitresult, ~] = fit( phiList, temp, ft, opts );
    
    pPhase2Temp(:,:,seq(ii)) = pPhase2Temp(:,:,seq(ii)).*exp(1i*(fitresult.b));
    
end

pPhase2 = sum(pPhase2Temp,3);
clear pPhase2Temp

figure(401), imagesc(abs(pPhase2)), axis square, colorbar
figure(402), imagesc(angle(pPhase2)), axis square, colorbar

% return

%% 5. Smoothing

[kx,ky] = meshgrid((1:sx2)-floor(sx2/2+1),(1:sy2)-floor(sy2/2+1));
kx = kx/fovX;
ky = ky/fovY;

% distList = -20:2:20; % rough pre-compensation of defocus aberration
distList = 0;

cc = zeros(length(distList),1);

for ii = 1:length(distList)
    
    approx = distList(ii)*(2*pi)*real(sqrt((1/lb).^2-(kx).^2-(ky).^2));
    cc(ii) = abs(sum(sum(exp(1i*approx).*exp(-1i*angle(pPhase2)).*~mk_ellipse(naX*0.8,naY*0.8,sx2,sy2))));
    
end

% figure(1), plot(distList,cc)
[~,cmax] = max(cc(:));

approx = distList(cmax)*(2*pi)*real(sqrt((1/lb).^2-(kx).^2-(ky).^2));
approx = approx - approx(floor(sy2/2+1),floor(sx2/2+1));

% dist = diff(f_dx);

% if max(dist) > 0
%     dist = dist(dist<0);
% else
%     dist = dist(dist>0);
% end

dist = mode(diff(f_dx)); %%%%%% peak distance (most frequent)
% dist = 4;
dist = (abs(mean(dist)))*2; % ideally 2 but practically 1.1 ~ 2.0

% adjust value based on how well low-pass filtered in Fig. 501

psf = fftshift(ifft2(ifftshift(pPhase2.*exp(-1i*approx))));

lpf = zeros(sy2,sx2);
lpf(floor(sy2/2+1)-floor(sy2/dist):floor(sy2/2+1)+ceil(sy2/dist), ...
    floor(sx2/2+1)-floor(sx2/dist):floor(sx2/2+1)+ceil(sx2/dist)) = 1;


[~, pIdx] = max(abs(psf(:)));
[pIdxY,pIdxX] = ind2sub(size(psf),pIdx);
pIdxX = pIdxX - floor(sx2/2+1);
pIdxY = pIdxY - floor(sy2/2+1);
lpf = circshift(lpf,[pIdxY,pIdxX]);

figure(500), imagesc(log(abs(psf))), axis square

psf = psf.*lpf; %% phase ramp unknown

figure(501), imagesc(log(abs(psf))), axis square

% psf = psf.*~mk_ellipse(sx2/dist,sy2/dist,sx2,sy2);
% psf = psf.*~mk_ellipse(naX*2/6,naY*2/6,sx2,sy2); % choose size based on fMask
% psf = psf.*~mk_ellipse(naX*2/5,naY*2/5,sx2,sy2); % choose size based on fMask

pPhase3Temp = fftshift(fft2(ifftshift(psf)));
pPhase3Temp = pPhase3Temp.*exp(1i*approx);

pPhase3 = pPhase3Temp;

%% # plot pupil after smoothing #######

% ccTemp = (naCond+naObj/15)/naObj; % additional range to plot
% ccTemp = (naCond+naObj/20)/naObj;
ccTemp = naCond/naObj;

temp = pPhase3;
temp = temp.*exp(-1i*angle(temp(floor(sy2/2+1),floor(sx2/2+1),1)));
temp = fftshift(ifft2(ifftshift(temp)));
temp = padarray(temp,[sy2,sx2],'both');
temp = fftshift(fft2(ifftshift(temp))).*exp(1i*0);

figure(502), imagesc(abs(temp)), axis off, axis square, colorbar

figure(503), imagesc(angle(temp).*~mk_ellipse(3*ccTemp*naX,3*ccTemp*naY,3*sx2,3*sy2),[-pi,pi])
axis square, axis off, colorbar, 
colormap(turbo) 
xlim([floor(1+floor(sx2/2+1)-ccTemp*naX)*3-5,ceil(floor(sx2/2+1)+ccTemp*naX)*3-3])
ylim([floor(1+floor(sx2/2+1)-ccTemp*naY)*3-3,ceil(floor(sx2/2+1)+ccTemp*naY)*3-3])

% return

%% 6. Validation

testIdx = 1;

pFinal = angle(pPhase3);

% naFactor = (naCond+naObj/15)/naObj;
naFactor = (naCond)/naObj;

E_o = circshift(fE(:,:,testIdx),[f_dy(testIdx),f_dx(testIdx)]);
E_o = E_o.*~mk_ellipse(naX*naFactor,naY*naFactor,sx2,sy2);

test = E_o.*exp(-1i*pFinal);
test = padarray(test,[sy2,sx2],'both');
test = circshift(test,[-f_dy(testIdx),-f_dx(testIdx)]);
test = fftshift(ifft2(ifftshift(test)));

test2 = padarray(E_o,[sy2,sx2],'both');
test2 = circshift(test2,[-f_dy(testIdx),-f_dx(testIdx)]);
test2 = fftshift(ifft2(ifftshift(test2)));

figure(601), 

aMax = mean(abs(test(:)))*2;

subplot(2,3,1), imagesc((abs(test)),[0,aMax]), colorbar, title('Corrected'), axis image, axis off
colormap(gca,'gray'), axis off

subplot(2,3,4), imagesc((abs(test2)),[0,aMax]), colorbar, title('Raw'), axis image, axis off
colormap(gca,'gray'), axis off

subplot(2,3,2), imagesc(unwrap2(angle(test))-mean2(unwrap2(angle(test)))), colorbar, title('Corrected'), axis square
% subplot(2,2,2), imagesc((angle(test)),[-pi,pi]), colorbar, title('Corrected'), axis image, axis off
colormap(gca,'parula'), axis off

subplot(2,3,5), imagesc(unwrap2(angle(test2))-mean2(unwrap2(angle(test2)))), colorbar, title('Raw'), axis square
% subplot(2,2,4), imagesc((angle(test2)),[-pi,pi]), colorbar, title('Raw'), axis image, axis off
colormap(gca,'parula'), axis off

subplot(2,3,3), imagesc(log(abs(fftshift(fft2(ifftshift(test)))))), axis image, axis off, colorbar
