function [input_field,output_field]=vector2scalarfield(input_field,output_field)

main_pol=1;
if size(input_field,3)==2
    dominant_pol=(mean(abs(input_field(:,:,:,:)),[1 2])>mean(abs(input_field(:,:,2,:)),[1 2]));
    dominant_pol(:,:,2,:)=~dominant_pol(:,:,1,:);
    main_pol=sum(input_field.*exp(-1i.*angle(mean(input_field.*dominant_pol,3))),[1 2]);
    main_pol=main_pol./sqrt(sum(abs(main_pol).^2,3));
end

input_field=squeeze(sum(input_field.*main_pol,3));
output_field=squeeze(sum(output_field.*main_pol,3));

end