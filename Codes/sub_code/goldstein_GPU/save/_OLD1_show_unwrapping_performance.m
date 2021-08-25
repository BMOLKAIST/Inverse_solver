function show_unwrapping_performance(unwrapped_phase)
[residue_map] = get_residue_gpu(unwrapped_phase);
[residue_idx,residue_value] = get_residu_position_values(residue_map);
figure;
subplot(1,3,1);
imagesc(unwrapped_phase); axis image;
subplot(1,3,2);
imagesc(residue_map); axis image;


dx=unwrapped_phase(1:size(unwrapped_phase,1)-1,2:size(unwrapped_phase,2))...
    -unwrapped_phase(2:size(unwrapped_phase,1),2:size(unwrapped_phase,2));
dy=unwrapped_phase(2:size(unwrapped_phase,1),1:size(unwrapped_phase,2)-1)...
    -unwrapped_phase(2:size(unwrapped_phase,1),2:size(unwrapped_phase,2));

fact=0.9999;
wrapping=((abs(dx)>=fact*pi)+(abs(dy)>=fact*pi))>1;

subplot(1,3,3);
imagesc(...
    wrapping...
    ); axis image;

figure; imagesc(residue_map); axis image;
end