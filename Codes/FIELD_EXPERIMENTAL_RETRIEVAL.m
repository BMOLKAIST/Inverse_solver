classdef FIELD_EXPERIMENTAL_RETRIEVAL < handle
    properties (SetAccess = private, Hidden = true)
        parameters;
        
        utility;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            %SIMULATION PARAMETERS
            params.resolution_image=[1 1]*0.1;
            params.use_abbe_correction=true;
            params.cutout_portion=1/3;
            params.other_corner=false;%false;%if the field is in another corner of the image
            params.conjugate_field=false;
            
            
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FIELD_EXPERIMENTAL_RETRIEVAL(params)
            h.parameters=params;
        end
        function [input_field,output_field,updated_optical_parameters]=get_fields(h,bg_file,sp_file, ROI)
            if sum(strfind(sp_file, 'tiff')) ~= 0 % Herve's tissue
                input_field=loadTIFF(bg_file);
                output_field=loadTIFF(sp_file);
            elseif sum(strfind(sp_file, '\set0')) ~= 0 % My setup data
                input_field=load_tiff_MS_setup(bg_file);
                output_field=load_tiff_MS_setup(sp_file);
            elseif sum(strfind(sp_file, '\0')) ~= 0 % CART data
                input_field=load_tomocube_PNG(bg_file);
                output_field=load_tomocube_PNG(sp_file);
            else % Bead droplet data is itself field. Will be converted in the main script.
                error('Filetype is not compatible.')
            end
            
            if ~isequal(size(input_field),size(output_field))
                error('Background and sample field must be of same size');
            end
            if size(input_field,1)~=size(input_field,2)
                error('the image must be a square');
            end
            if h.parameters.resolution_image(1)~=h.parameters.resolution_image(2)
                error('please enter an isotropic resolution for the image');
            end
            if h.parameters.resolution(1)~=h.parameters.resolution(2)
                error('please enter an isotropic resolution for the output image');
            end
            
            if nargin == 4
                input_field = input_field(ROI(1):ROI(2), ROI(3):ROI(4),:,:,:);
                output_field = output_field(ROI(1):ROI(2), ROI(3):ROI(4),:,:,:);
            end
            
            
            input_field=fftshift(fft2(ifftshift(input_field)));
            output_field=fftshift(fft2(ifftshift(output_field)));
            
            %1 center the field in the fourier space
            delete_band_1=round(size(input_field,1).*h.parameters.cutout_portion):size(input_field,1);
            delete_band_2=round(size(input_field,2).*h.parameters.cutout_portion):size(input_field,2);
            if h.parameters.other_corner
                delete_band_2=1:round(size(input_field,2).*(1-h.parameters.cutout_portion));
            end
            normal_bg=input_field(:,:,1);
            normal_bg(delete_band_1,:,:)=0;
            normal_bg(:,delete_band_2,:)=0;
            
            [center_pos_1,center_pos_2]=find(abs(normal_bg)==max(abs(normal_bg(:))));
            
            input_field=fftshift(fftshift(circshift(input_field,[1-center_pos_1,1-center_pos_2,0]),1),2);
            output_field=fftshift(fftshift(circshift(output_field,[1-center_pos_1,1-center_pos_2,0]),1),2);
            
            %2 match to the resolution
            old_side_size=size(input_field,1);
            resolution_ratio=h.parameters.resolution(1)/h.parameters.resolution_image(1);
            if resolution_ratio>=1
                %crop
                crop_size=round((1/2)*(size(input_field,1)-size(input_field,1)./resolution_ratio));
                input_field=input_field(1+crop_size:end-crop_size,1+crop_size:end-crop_size,:);
                output_field=output_field(1+crop_size:end-crop_size,1+crop_size:end-crop_size,:);
            end
            if resolution_ratio<1
                %padd
                padd_size=-round((1/2)*(size(input_field,1)-size(input_field,1)./resolution_ratio));
                input_field=padarray(input_field,[padd_size padd_size 0],'both');
                output_field=padarray(output_field,[padd_size padd_size 0],'both');
            end
            h.parameters.resolution(1)=h.parameters.resolution_image(1).*old_side_size./size(input_field,1);
            h.parameters.resolution(2)=h.parameters.resolution_image(2).*old_side_size./size(input_field,2);
            %crop the NA
            warning('off','all');
            h.parameters.size(1)=size(input_field,1);
            h.parameters.size(2)=size(input_field,2);
            
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            
            
            input_field=input_field.*h.utility.NA_circle;
            output_field=output_field.*h.utility.NA_circle;
            
            
            
            
            input_field=fftshift(ifft2(ifftshift(input_field)));
            output_field=fftshift(ifft2(ifftshift(output_field)));
            if h.parameters.conjugate_field
                input_field=conj(input_field);
                output_field=conj(output_field);
            end
            input_field=reshape(input_field,size(input_field,1),size(input_field,2),1,[]);
            output_field=reshape(output_field,size(output_field,1),size(output_field,2),1,[]);
            input_field = input_field(3:(end-2),3:(end-2),:,:,:);
            output_field = output_field(3:(end-2),3:(end-2),:,:,:);
            a = sum(sum(abs(input_field),1),2) / (size(input_field,1)*size(input_field,2));
            input_field = input_field ./ a;
            output_field = output_field ./ a;
            
            h.parameters.size(1)=size(input_field,1);
            h.parameters.size(2)=size(input_field,2);
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            
            warning('abbe cos coefficient');
            updated_optical_parameters=h.parameters;
            base_param=BASIC_OPTICAL_PARAMETER();
            warning ('off','all');
            updated_optical_parameters=update_struct_no_new_field(base_param,updated_optical_parameters);
            warning ('on','all');
        end
    end
end

function Data=load_tiff_MS_setup(namee)
cd(namee)
spdir=dir('*.tiff');
for i1=1:length(spdir)
    img=imread(spdir(i1).name);
    if i1==1
        Data=zeros([size(img) length(spdir)]);
    end
    Data(:,:,i1)=img;
end

end
function Data=load_tomocube_PNG(namee)
cd(namee)
spdir=dir('*.png');
for i1=1:length(spdir)
    img=imread(spdir(i1).name);
    if i1==1
        Data=zeros([size(img) length(spdir)]);
    end
    Data(:,:,i1)=img;
end

end
function object=loadTIFF(fname,num_images)
info = imfinfo(fname);
if nargin == 1
num_images = numel(info);
end
display(['Number of images (read) : ',num2str(num_images)]);
object= zeros(info(1).Width,info(1).Height,num_images,'single');
for k = 1:num_images
   object(:,:,k) = imread(fname, k, 'Info', info);
end
if num_images==1
   object=squeeze(object);
end
end